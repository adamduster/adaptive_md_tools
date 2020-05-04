#!/usr/bin/env python
"""
This file contains the 'core loop' subroutine which iterates over all of the
frames in the trajectory and is the main driver of the program.

It currently also contains all the logic for printing out adaptive partitioning
files, and for handling the indicator.

TODO:
Move indicator subroutines to their own file to clean up code.
"""
__author__ = 'Adam Duster'
__copyright__ = ''
__credits__ = ['Adam Duster']
__license__ = 'CC-BY-SA'
__version__ = '0.1'
__email__ = 'adam.duster@ucdenver.edu'
__status__ = 'Development'
import os
import MDAnalysis as mda
from .indicator import *
from .indicator_mda_selections import *
from adaptive_md_tools.ap_algorithms import write_partitions, write_sispa
import adaptive_md_tools.mdtools as mdtools

import sys
import gc


def core_loop(keywords, indi):
    """
    Main loop for processing indicator. Here, the indicator is calculated if
    requested. The atoms can be translated such that a specific atom or the
    indicator is in the center of the box if requested.

    1. Set up universe
    2. Set up groups

    3. Main Loop

    :param keywords: input keywords
    :type keywords: dict
    :param indi: indicator classz
    :type indi: Indicator
    """
    keywords['wrap_style'] = 'residues'
    # mda.core.flags['use_periodic_selections'] = True
    # mda.core.flags['use_KDTree_routines'] = False

    # Main universe with original coordinates
    try:
        u = initialize_universe(keywords["topology"], keywords["in_coords"],
                                *keywords["dimensions"], 0)
    except:
        sys.exit("Error loading initial universe")
    # This is the modified universe for the coordinate output
    if not keywords['mcec']:
        all_u = add_indicator_to_universe(u)
    else:
        all_u = add_indicator_to_universe(u, natoms=2)

    sels = setup_selection(u, all_u, indi, keywords)

    # Setup the initial translation vector
    if keywords["wrap"]:
        box_center = u.dimensions[:3] / 2.0

    #Setup the groups
    if "groups_path" in keywords:
        groups_file = mdtools.GroupsFile(keywords["groups_path"])
        groups = groups_file.getGroups()
        ap = True

    #Setup the elements
    if keywords["write_partitions"] or keywords["write_sispa"]:
        try:
            elements = mdtools.get_elements(keywords["elements_file"],
                                            keywords["elements_file_type"])
        except:
            print("Error reading elements. Was the elements file specified?")
            sys.exit()
        if elements.size != u.atoms.n_atoms:
            print("Error, number of elements does not match with system # els")
            sys.exit()
    if keywords["write_partitions"]:
        if not os.path.isdir(keywords["write_folder"]):
            try:
                os.mkdir(keywords["write_folder"])
            except OSError:
                sys.exit("Error making partitions directory")
    if keywords["write_sispa"]:
        if not os.path.isdir(keywords["sispa_folder"]):
            try:
                os.mkdir(keywords["sispa_folder"])
            except OSError:
                sys.exit("Error making partitions directory")
    if keywords["out_coords"]:
        if keywords["ind_method"] > -1:
            try:
                W = mda.Writer(keywords["out_coords"], sels.all.n_atoms)
            except IOError:
                print("Error opening output coord file: " + keywords["out_coords"])
                sys.exit('Error opining trajectory out with indicator')
        else:
            try:
                W = mda.Writer(keywords["out_coords"], sels.sys.n_atoms)
            except IOError:
                print("Error opening output coord file: " + keywords["out_coords"])
                sys.exit('Error opining trajectory out with indicator')

    # Main Loop
    nsteps = len(u.trajectory)
    for t in range(nsteps):
        ts = u.trajectory[t]

        # Initialize the selections for this timestep
        sels.set_dah(u, indi.donor)
        sels.reset_universe_selections(u, all_u)
        donor_coords = sels.d.positions

        #  Wrapping instructions
        if keywords["wrap"]:
            # Set the dimensions in the systems for this timestep
            ts.dimensions[:] = [*keywords['dimensions'], 90, 90, 90]
            u.dimensions[:] = [*keywords['dimensions'], 90, 90, 90]
            all_u.dimensions[:] = [*keywords['dimensions'], 90, 90, 90]

            # Translate the no-indicator system such that the donor is in the
            # center of the box
            translation_vector = box_center - donor_coords
            sels.sys.translate(translation_vector)

            # Wrap atoms into the box
            sels.sys.wrap(compound=keywords['wrap_style'], center='com')

        if keywords["ind_method"] == -1:
            if keywords["out_coords"]:
                W.write(sels.sys)
            indi.x_i = box_center # Reset this for print_partitions to pass as center
        elif keywords["ind_method"] > -1:
            if not keywords["allow_hop"]:
                if ts.frame % keywords["write_freq"] != 0:
                    continue
            if keywords["ratio_topology_change"]:
                ratio_topology_change(u, indi, sels, keywords)
                if indi.hop and keywords["allow_hop"]:
                    u, all_u = do_hop(u,
                                      all_u,
                                      indi,
                                      ts,
                                      keywords,
                                      sels,
                                      groups,
                                      intra=True)

            calc_indicator(u, all_u, indi, sels, keywords)

            if keywords["wrap"]:
                # Now translate the indicator into the center of the box and wrap
                ind_translate = box_center - indi.x_i
                translation_vector += ind_translate
                # The below wrap commands are useful to output a debug trajectory.
                sels.all.translate(ind_translate)
                sels.all.wrap(compound=keywords['wrap_style'], center='com')

            # Update the topology and AP groups
            if indi.hop and keywords["allow_hop"]:
                u, all_u = do_hop(u, all_u, indi, ts, keywords, sels, groups)
                if keywords["wrap"]:
                    sels.sys.translate(translation_vector)
                    sels.sys.wrap(compound=keywords['wrap_style'], center='com')
                    sels.all.translate(translation_vector)
                    sels.all.wrap(compound=keywords['wrap_style'], center='com')

            if keywords["out_coords"]:
                W.write(sels.all)

        # Write the coordinates for sispa
        if keywords["write_sispa"] and\
                ts.frame % keywords["write_freq"] == 0:
            write_sispa(groups, ts, elements, indi.x_i, keywords)

        # Output the xyz coordinates if we are making xys for AP
        if keywords["write_partitions"] and\
                ts.frame % keywords["write_freq"] == 0:
            write_partitions(groups, ts, elements, indi.x_i, keywords)

        # End of loop
    return


def setup_selection(u, all_u, indi, keywords):
    """
    Set up the selections for the system

    Parameters
    ----------
    u: MDAnalysis.Universe
        main universe object
    all_u: MDAnalysis.Universe
        universe with proton
    indi: Indicator class
        proton indicator
    keywords: dict
        program parameters

    Returns
    -------

    """
    # Selection setup
    acceptor_types = list(indi.rxh.keys())
    proton_types = keywords["proton_types"]
    if keywords["ind_method"] == -1:
        proton_types = ["HT"]
        acceptor_types =["OT"]
        sels = SelectionsInd1(u,
                              all_u,
                              proton_types,
                              acceptor_types,
                              1.0,
                              donor_index=indi.donor)
    elif keywords["ind_method"] in [1]:
        sels = SelectionsInd1(u,
                              all_u,
                              proton_types,
                              acceptor_types,
                              indi.rlist,
                              donor_index=indi.donor)
    elif keywords["ind_method"] == 2:
        sels = SelectionsInd2(u,
                              all_u,
                              proton_types,
                              acceptor_types,
                              indi.rlist,
                              donor_index=indi.donor)
    else:
        sels = Selections(u,
                          all_u,
                          proton_types,
                          acceptor_types,
                          indi.rlist,
                          donor_index=indi.donor)
    if keywords["ind_method"] in [0]:
        sels.set_donor = sels.set_one_donor
        sels.set_acc = sels.set_original_acc
    return sels


def get_group_geom_center(u, types, ind, redundant=True, return_types=False):
    """
    Get the center of geometry for the group that index in ind is a member of.

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe that has the atoms which we want to get the com of
    types: list of str
        List of atom types for each ind
    ind: list ints
        0-Based indices to get the group center of mass from.
    redundant: bool
        Not sure
    return_types: bool
        If True, return a tuple containing the type of the

    Returns
    -------
    com: list of float nd.arrays
    ret_types: Optional, list of str
       The types of the first atom in the selection for the COG.

    for return_types=True, the tuple is:
    (com, ret_types)

    Warnings
    --------
    This subroutine frequently has issues.
    """
    com = []
    ret_types = []
    num_types = len(types)
    type_str = '(type %s' % (types[0])
    for i in range(1, num_types):
        type_str += " or type %s" % types[i]
    type_str += ')'

    for i in range(len(ind)):
        sel_str = ('same resnum as bynum %d' % (ind[i] + 1))
        try:
            sel = u.select_atoms(sel_str)
            sel = sel.select_atoms(type_str)
        except:
            "Error selecting atoms"
            sys.exit()
        if sel.n_atoms < 1:
            sys.exit("Empty selection for group COM")
        if redundant:
            com.append(sel.center_of_geometry())
        else:
            found = False
            for c in com:
                if np.isclose(c, sel.center_of_geometry()).all():
                    found = True
            if not found:
                com.append(sel.center_of_geometry())
                ret_types.append(sel.types[0])
    if return_types:
        return com, ret_types
    else:
        return com


def get_group_ind(u, types, ind, redundant=True):
    """
    Return a list of indices for atoms in COG groups for indicator 1.

    I don't quite understand the code anymore...

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe that has the atoms which we want to get the com of
    types: list of str
        List of atom types for each ind
    ind: list of lists of ints
        Contains atoms in each group
    redundant: bool
        Not sure

    Returns
    -------
    com: list of float nd.arrays

    TODO
    ----
    Figure out what the hell this does.

    Warnings
    --------
    This subroutine frequently has issues.
    """
    inds = []
    num_types = len(types)
    type_str = '(type %s' % (types[0])
    for i in range(1, num_types):
        type_str += " or type %s" % types[i]
    type_str += ')'

    for i in range(len(ind)):
        sel_str = ('same resnum as bynum %d' % (ind[i] + 1))
        try:
            sel = u.select_atoms(sel_str)
            sel = sel.select_atoms(type_str)
        except:
            "Error selecting atoms"
            sys.exit()
        if sel.n_atoms < 1:
            sys.exit("Empty selection for group COM")
        if redundant:
            inds.append(sel.indices[0])
        else:
            found = False
            for c in inds:
                if np.isclose(c, sel.indices[0]):
                    found = True
            if not found:
                inds.append(sel.indices[0])
    return inds


def sel_type_and_same_res(u, types, ind):
    """
    Generic subroutine to select atoms with a type given in the list 'types'
    with the same resnum as the index of ind

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe with atoms for selection
    types: list of str
        List of atom types for each ind
    ind: integer
        1-based index of atom with resnum.

    Returns
    -------
    sel: MDAnalysis.selection

    TODO
    ----
    This subroutine contains logic that is used throughout the code. That
    could all be consolidated.
    """
    num_types = len(types)
    type_str = '(type %s' % (types[0])
    for i in range(1, num_types):
        type_str += " or type %s" % types[i]
    type_str += ')'
    sel_str = ('same resnum as bynum %d' % (ind[i] + 1))
    sel = u.select_atoms(sel_str)
    sel = sel.select_atoms(type_str)
    return sel


def check_for_donor_switch(u, indi, sels):
    """
    Check to see if an intramolecular proton transfer has occured.

    Compare the distances between the protons and protonatable sites of a
    donor group. If a proton is closer to another site than the one that
    it is bonded to, add the hop to the indi.hop list.f

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe with atoms for selection
    indi: Indicator class object.
        Has info about donors, acceptors, etc
    sels: Selection class object.
        Contains selections that correspond to the indi class

    Returns
    -------

    """
    largest_p = 0
    don_ind = 0
    sels.set_donors(u, indi.donor)
    d_inds = sels.d.indices.copy()
    d_pos = sels.d.positions.copy()

    # Iterate over all protonatable sites.
    for k in range(d_inds.size):
        # Set the donor to site k
        sels.set_dah(u, d_inds[k] + 1)
        for j in range(sels.a.positions.shape[0]):
            for m in range(sels.h.positions.shape[0]):
                # Compute the projection of the proton m onto the vector
                # between sites k and j
                rdh = sels.h.positions[m, :] - d_pos[k, :]
                rda = sels.a.positions[j, :] - d_pos[k, :]
                p = np.dot(rdh, rda) / np.linalg.norm(rda)**2

                # We will only select the largest rho in the case of two
                # simultaneous hops.
                if p > largest_p:
                    don_ind = d_inds[k]
                    largest_p = p
    # If there is no hop, reset the selection to the donor that we came in with
    if largest_p <= 0:
        sels.set_dah(u, indi.donor)
        return
    # If there is a hop, note that and switch the donor.
    if don_ind != indi.donor - 1:
        print("Swapping donors in residue based on H-bond dists at step: ",
              u.trajectory.frame)
        print("Old Don: {0}   New Don: {1}".format(indi.donor, don_ind + 1))
        indi.donor = don_ind + 1
    sels.set_dah(u, indi.donor)
    return


def calculate_weighted_coords(u, sels):
    """
    Calculate a weighted coordinate for a group of donors in the system based
    on the projection vectors between them and surrounding accpetors.

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe with atoms for selection
    sels: Selection class object.
        Contains selections that correspond to the indicator class

    Returns
    -------
    ndarray of floats with shape (3)

    Raises
    ------
    DivideByZero : The sum of the weights is 0 and thus unnormalizable

    Notes
    -----
    Here we calculate the vector between the donor K and all acceptors J. For
    each proton M, we calculate the projection onto the J-K vector and get
    the normalized distance of the projection D. We then weight each donor's
    location based on the sum of D's for each donor and return the weighted
    coordinate.

    The idea here is that if a proton is further from a given donor, it is
    more likely to transfer. We should weight the location of the indicator
    towards that donor.
    """
    d_inds = sels.d.indices.copy()
    d_pos = sels.d.positions.copy()
    num_d = d_inds.size
    w_d = np.zeros(num_d)

    for k in range(num_d):
        sels.set_dah(u, d_inds[k] + 1)
        for j in range(sels.a.positions.shape[0]):
            for m in range(sels.h.positions.shape[0]):
                rdh = sels.h.positions[m, :] - d_pos[k, :]
                rda = sels.a.positions[j, :] - sels.h.positions[m, :]
                p = np.dot(rdh, rda) / np.linalg.norm(rda)**2
                if p > 0:
                    p = 0
                w_d[k] += p
    total = np.sum(w_d)
    if total <= 0:
        print("Error weighting coordinates. There are probably no protons"
              " bound to the donor group")
        raise ZeroDivisionError
    w_d /= np.sum(w_d)
    don_com = w_d.reshape(num_d, 1) * d_pos
    return np.sum(don_com, axis=0)


def ratio_topology_change(u, indi, sels, keywords):
    """
    Adds a proton hop from one donor to the next to the indicator object
    depending on the D-H bond lengths for different atoms in the system.

    The proton is transferred if:
    r_km / [r_km + r_jm] > 0.5 (the distance between the donor and it's proton
                                is greater than an acceptor and the proton)

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe with atoms for selection
    indi: Indicator class object
        The indicator info for checking the transfer
    sels: Selection class object
        Contains selections that correspond to the indicator class
    keywords: dictionary
        Dictionary object which contains indicator method we are using.

    Notes
    -----
    This can be used to describe tautomerization proton transfer reactions such
    as that for the proton between the two oxygens in glutamic acid.

    TODO
    ----
    Remove dependency on keywords variable
    """
    if keywords["ind_method"] == 2:
        sels.set_donors(u, indi.donor)
    num_d = len(sels.d.indices)
    if num_d == 1:
        return
    indi.hop = []

    #Iterate over all donor-acceptor combinations
    for k in range(num_d):
        for j in range(num_d):
            if j == k:
                continue
            sels.set_proton(u, sels.d.indices[k] + 1)
            num_h = len(sels.h.indices)
            for m in range(num_h):

                # Standard rho calculation
                rdh = sels.d.positions[k, :] - sels.h.positions[m, :]
                rdh = np.linalg.norm(rdh)
                rah = sels.d.positions[j, :] - sels.h.positions[m, :]
                rah = np.linalg.norm(rah)
                p = rdh / (rdh + rah)
                if p > 0.5:
                    indi.hop.append((m, j, p, k))
    sels.set_dah(u, indi.donor)
    return


def calc_indicator(u, all_u, indi, sels, keywords):
    """
    Stage and calculate the location of the indicator.
    Different things happen depending on the indicator method.

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe with atoms for selection
    all_u: MDAnalysis.universe
        Universe with indicator atom appended to it
    indi: Indicator class object
        The indicator info for checking the transfer
    sels: Selection class object
        Contains selections that correspond to the indicator class
    keywords: dictionary
        Dictionary object which contains indicator method we are using.

    Returns
    -------

    """
    # Just set the indicator to the donor if there is no indicator
    if keywords["ind_method"] == -1:
        indi.x_i = sels.d.positions[0]
        return
    # Set up atom selections for donor and acceptors
    sels.set_dah(u, indi.donor)

    # Print out what the hydrogens, acceptors, donors are
    if keywords["indicator_verbose"]:
        print("#**************** STEP %d" % u.trajectory.frame)
        print("Donor indicies in order: ", sels.d.indices)
        print("Acc indices: ", sels.a.indices)
        print("Hyd indices: ", sels.h.indices)
    # Calculate the indicator location
    if keywords["ind_method"] == 0:
        indi.calc_indicator(sels.d.positions[0], sels.a.positions,
                            sels.h.positions, sels.d.types[0], sels.a.types)
    elif keywords["ind_method"] == 1:
        don_com = get_group_geom_center(u, list(indi.rxh.keys()),
                                        [sels.d.indices[0]])[0]
        acc_com, atypes = get_group_geom_center(u,
                                                list(indi.rxh.keys()),
                                                sels.a.indices,
                                                redundant=False,
                                                return_types=True)
        acc_com = np.asarray(acc_com)

        indi.calc_indicator(don_com, acc_com, sels.h.positions, sels.d.types[0],
                            atypes)
    elif keywords["ind_method"] == 2:
        check_for_donor_switch(u, indi, sels)
        indi.calc_indicator(sels.d.positions[0], sels.a.positions,
                            sels.h.positions, sels.d.types[0], sels.a.types)
    elif keywords["ind_method"] == 3:
        don_com = [calculate_weighted_coords(u, indi, sels)]
        acc_com = get_group_geom_center(u, list(indi.rxh.keys()),
                                        sels.a.indices)
        h_coords = []
        for i in range(sels.d.n_atoms):
            sels.set_proton(u, sels.d.indices[i] + 1)
            h_coords.append(sels.h.positions)

        indi.calc_indicator(sels.d.positions, sels.a.positions, h_coords,
                            sels.d.types, sels.a.types, don_com, acc_com)
    elif keywords["ind_method"] in [4, 6, 7, 9, 11]:
        don_com = get_group_geom_center(u, list(indi.rxh.keys()),
                                        [sels.d.indices[0]])
        acc_com = get_group_geom_center(u, list(indi.rxh.keys()),
                                        sels.a.indices)
        h_coords = []
        for i in range(sels.d.n_atoms):
            sels.set_proton(u, sels.d.indices[i] + 1)
            h_coords.append(sels.h.positions)

        indi.calc_indicator(sels.d.positions, sels.a.positions, h_coords,
                            sels.d.types, sels.a.types, don_com, acc_com)

    elif keywords["ind_method"] in [8]:
        don_com = [calculate_weighted_coords(u, indi, sels)]
        acc_com = get_group_geom_center(u, list(indi.rxh.keys()),
                                        sels.a.indices)
        acc_com = np.asarray(acc_com)
        indi.calc_indicator(don_com, acc_com, sels.h.positions, sels.d.types[0],
                            sels.a.types)
    if keywords["mcec"]:
        accs = sels.set_all_acc(u)
        hyds = sels.set_all_protons(u)
        group_locs = None
        if indi.correction_groups:
            group_locs = get_group_positions(u, indi)
        indi.calc_mcec(hyds.positions, accs.positions, accs.types, group_locs)

    # Update the atoms positions in the larger universe
    if keywords["mcec"]:
        all_u.trajectory.coordinate_array[0, :-2] = sels.sys.positions[:]
        all_u.trajectory.coordinate_array[0, -2] = indi.x_i
        all_u.trajectory.coordinate_array[0, -1] = indi.x_mcec
    else:
        all_u.trajectory.coordinate_array[0, :-1] = sels.sys.positions[:]
        all_u.trajectory.coordinate_array[0, -1] = indi.x_i
    return


def get_group_positions(u, indi):
    """
    Return the locations of the members in the correction groups.

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe with atoms for selection
    indi: Indicator class object
        The indicator info for checking the transfer

    Returns
    -------
    list of ndarrays with shape (n,3)

    """
    positions = []
    for i in indi.correction_groups:
        selstr = 'bynum %d' % i[0]
        for j in i[1:]:
            selstr += ' or bynum %d' % j
        positions.append(u.select_atoms(selstr).positions)
    return positions


def do_hop_ind_class(u, indi, sels, intra, keywords):
    """
    Here we adjust the indicator class for a proton hop. We change the donors
    and acceptors, and find the transferring hydrogen. Then we reset the
    selections.

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe that has the atoms which we want to get the com of
    indi: Indicator class object
        The indicator info for checking the transfer
    sels: Selection class object
        Contains selections that correspond to the indicator class
    keywords: dictionary
        Dictionary object which contains indicator method we are using.
    intra: bool
        Whether this is an intramolecular proton hop or not

    Returns
    -------
    transfer_h: int
        0-based atom index of hydrogen that is transfered.
    old_don: int
        0-based atom index of donor before transfer
    new_don: int
        0-based atom index of donor after transfer
    """
    # Choose the correct hop if there are multiple
    if len(indi.hop) > 1:
        print("Warning: multiple hops at frame ", u.trajectory.frame)
        print("Choosing largest pmj")
        for i in range(1, len(indi.hop)):
            if indi.hop[i][2] > indi.hop[0][2]:
                indi.hop[0] = indi.hop[i]

    # ???
    if keywords["ind_method"] == 7:
        if indi.hop[0][4] == True:
            intra = True

    # We need to look for the new index from the donor selection rather than
    # the acceptor if its an intramolecular hop
    if intra:
        if keywords["ind_method"] == 2:
            sels.set_donors(u, indi.donor)
        new_don = sels.d.indices[indi.hop[0][1]]
    else:
        new_don = sels.a.indices[indi.hop[0][1]]

    # Choose the old donor from the list of multiple donors.
    if keywords["ind_method"] in [3, 4, 6, 7, 9, 11]:
        old_don = sels.d.indices[indi.hop[0][3]]
        #
    elif keywords["ind_method"] in [1]:
        inds = get_group_ind(u,
                             list(indi.rxh.keys()),
                             sels.a.indices,
                             redundant=False)
        new_don = inds[indi.hop[0][1]]
        old_don = indi.donor - 1
    else:  # For original indicator
        old_don = indi.donor - 1
    sels.set_proton(u, old_don + 1)
    transfer_h = sels.h.indices[indi.hop[0][0]]
    indi.donor = new_don + 1

    return transfer_h, old_don, new_don


def do_hop(u, all_u, indi, ts, keywords, sels, groups, intra=False):
    """
    This subroutine has a lot going on. We hop the proton by:
        1. Changing the donor in the indicator and resetting selections
        2. Change the topology
        3. Because we cannot change the topology for the whole trajectory, we
        must delete u and all_u, then reinitialize them
    based on the new topologies.

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe with atoms for selection
    all_u: MDAnalysis.universe
        Universe with indicator atom appended to it
    indi: Indicator class object
        The indicator info for checking the transfer
    translation_vector:
        Vector to translate system by before wrapping.
    ts: MDAnalysis.universe.trajectory.frame
        The current trajectory frame
    keywords: dictionary
        Dictionary object which contains indicator method we are using.
    sels: Selection class object
        Contains selections that correspond to the indicator class
    intra: bool
        Whether this is an intramolecular proton hop or not
    """
    # Fix the indicator class and get the indices of the atoms involved in the
    # transfer
    frame = ts.frame
    transfer_h, old_don, new_don = do_hop_ind_class(u, indi, sels, intra,
                                                    keywords)
    if keywords["verbose"]:
        print("Proton hop at frame ", ts.frame, "Donor: ", old_don + 1, "ACC:",
              new_don + 1)
    #
    # Change groups of atoms
    groups.transferAtom(transfer_h, new_don)
    #
    # Go through and get the new list of bonds for rewriting the topology
    new_bonds = calc_new_bonds_from_u(u, transfer_h, new_don)
    #
    # Write the new psf in a confusing and horrible way
    if keywords["topology_type"] == 'psf':
        tmp_fi = 'tmp.psf'
        fi = mdtools.PSFFile(keywords["topology"])
        topo = fi.getTopo()
        topo.mm_types.resId[:] = groups.atomGroups[:]
        tmp_fi = 'tmp.psf'
        topo.set_ofi(tmp_fi)
        topo.write_psf(new_bonds, u.atoms.n_atoms)
    elif keywords["topology_type"] == 'mol2':
        fi = mdtools.Mol2File(keywords["topology"])
        topo = fi.getTopo()
        topo.mm_types.resId[:] = groups.atomGroups[:]
        tmp_fi = 'tmp.mol2'
        topo.set_ofi(tmp_fi)
        topo.write_mol2(new_bonds)
    #
    # Reinitialize the universe
    del u
    del all_u
    gc.collect()
    try:
        new_u = initialize_universe(tmp_fi,
                                    keywords["in_coords"],
                                    *keywords["dimensions"],
                                    frame=frame)
    except:
        sys.exit("Error reloading universe after proton hop")
    if keywords['mcec']:
        new_all_u = add_indicator_to_universe(new_u, frame=frame, natoms=2)
    else:
        new_all_u = add_indicator_to_universe(new_u, frame=frame)
    sels.reset_all(new_u, new_all_u, indi.donor)
    indi.reset_hop()

    return new_u, new_all_u


def calc_new_bonds_from_u(u, transfer_h, new_don):
    """
    Get the new bond list by deleting the bond between old_don and transfer_h,
    and adding the one between new_don and transfer_h.

    Here we make the assumption that the transfer H is only
    bonded to one atom. We just switch that bond

    Parameters
    ----------
    u: MDAnalysis.universe
        Main universe with atoms for selection
    transfer_h: int
        0-based atom index of hydrogen that is transfered.
    old_don: int
        0-based atom index of donor before transfer
    new_don: int
        0-based atom index of donor after transfer

    Returns
    -------
    new_bonds: list of list of ints
        List of bonds. Bonds are lists with a pair of ints.
        Example: [[1,2],[3,4]]
    """
    num_bonds = u.bonds.indices.shape[0]
    new_bonds = []
    found_bond = False
    for b in range(num_bonds):
        if transfer_h in u.bonds.indices[b, :2]:
            new_bonds.append((new_don, transfer_h))
            if found_bond == False:
                found_bond = True
            else:
                print("fatal error")
        else:
            new_bonds.append((u.bonds.indices[b, 0], u.bonds.indices[b, 1]))
    if len(new_bonds) != num_bonds:
        sys.exit("Error reorganizing bonds")
    return new_bonds


def add_indicator_to_universe(u,
                              ind_type="IND",
                              ind_name="IND",
                              frame=0,
                              natoms=1):
    """
    Add another atom and return the current universe

    Parameters
    ----------
    u: MDAnalysis.Universe
        Universe to add indicator to
    natoms: int
        The number of atoms to append to the universe.

    Returns
    -------
    all_u: MDAnalysis.Universe
        The new universe
    """
    all_atoms = u.select_atoms("all")
    ind_sel = u.select_atoms("bynum %d" % 1)
    all_u = mda.Merge(all_atoms, ind_sel)
    for i in range(1, natoms):
        all_atoms = all_u.select_atoms("all")
        ind_sel = u.select_atoms("bynum %d" % 1)
        all_u = mda.Merge(all_atoms, ind_sel)
    all_u.dimensions = u.dimensions
    return all_u


def initialize_universe(struct, coords, xdim, ydim, zdim, frame=0):
    """
    Initialize an MDAnalysis universe

    Parameters
    ----------
    struct: str
        The name of the structure file to input to the universe
    coords: str
        The name of the coordinate file for the universe
    frame: int
        timestep to jump to
    xdim: float
        x dimension in A
    ydim: float
        y dimension in A
    zdim: float
        z dimension in A
    :return: universe
    """
    try:
        u = mda.Universe(struct, coords)
    except IOError:
        print("Error loading MDA universe from coordinates and structure",
              struct, coords)
        raise
    try:
        u.trajectory[frame]
    except LookupError:
        print("Error going to trajectory frame", frame)
        raise

    u.dimensions = [xdim, ydim, zdim, 90, 90, 90]
    return u
