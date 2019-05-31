#!/usr/bin/env python
"""

"""
__author__ = 'Adam Duster'
__copyright__ = ''
__credits__ = ['Adam Duster']
__license__ = 'CC-BY-SA'
__version__ = '0.1'
__email__ = 'adam.duster@ucdenver.edu'
__status__ = 'Development'
from scipy.spatial.distance import *
import os
import MDAnalysis as mda
from .indicator import *
from .indicator_mda_selections import *
import  adaptive_md_tools.mdtools as mdtools
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
    :param indi: indicator class
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

    # Selection setup
    acceptor_types = list(indi.rxh.keys())
    proton_types = keywords["proton_types"]

    if keywords["ind_method"] in [1]:
        sels = SelectionsInd1(u, all_u, proton_types, acceptor_types,
                              indi.rlist, donor_index=indi.donor)
    elif keywords["ind_method"] == 2:
        sels = SelectionsInd2(u, all_u, proton_types, acceptor_types,
                              indi.rlist, donor_index=indi.donor)
    else:
        sels = Selections(u, all_u, proton_types, acceptor_types,
                          indi.rlist, donor_index=indi.donor)
    if keywords["ind_method"] in [0]:
        sels.set_donor = sels.set_one_donor
        sels.set_acc = sels.set_original_acc

    # Setup the initial translation vector
    if keywords["wrap"]:
        box_center = u.dimensions[:3] / 2.0

    #Setup the groups
    if "groups_path" in keywords:
        groups_file = mdtools.GroupsFile(keywords["groups_path"])
        groups = groups_file.getGroups()
        ap = True

    #Setup the elements
    if keywords["write_partitions"]:
        try:
            elements = mdtools.get_elements(keywords["elements_file"],
                                     keywords["elements_file_type"])
        except:
            print("Error reading elements. Was the elements file specified?")
            sys.exit()
        if elements.size != u.atoms.n_atoms:
            print("Error, number of elements does not match with system # els")
            sys.exit()
        if not os.path.isdir(keywords["write_folder"]):
            try:
                os.mkdir(keywords["write_folder"])
            except OSError:
                sys.exit("Error making partitions directory")
    try:
        W = mda.Writer(keywords["out_coords"], sels.all.n_atoms)
        # Wdebug = mda.Writer('debug.dcd', sels.sys.n_atoms)
    except IOError:
        print("Error opening output coord file: " + keywords["output"])

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

        if not keywords["allow_hop"]:
            if ts.frame % keywords["write_freq"] != 0:
                continue
        if keywords["ratio_topology_change"]:
            ratio_topology_change(u, indi, sels, all_u, ts, keywords, groups)
            if indi.hop and keywords["allow_hop"]:
                u, all_u = do_hop(u, all_u, indi, ts, keywords,
                       sels, groups, intra=True)

        calc_indicator(u, all_u, indi, sels, keywords, groups, ts)

        if keywords["wrap"]:
            # Now translate the indicator into the center of the box and wrap
            ind_translate = box_center - indi.x_i
            translation_vector += ind_translate
            # The below wrap commands are useful to output a debug trajectory.
            sels.all.translate(ind_translate)
            sels.all.wrap(compound=keywords['wrap_style'], center='com')

        # Wdebug.write(sels.sys)
        W.write(sels.all)

        # Update the topology and AP groups
        if indi.hop and keywords["allow_hop"]:
            u, all_u = do_hop(u, all_u, indi, ts, keywords,
                   sels, groups)
            if keywords["wrap"]:
                sels.sys.translate(translation_vector)
                sels.sys.wrap(compound=keywords['wrap_style'], center='com')
                sels.all.translate(translation_vector)
                sels.all.wrap(compound=keywords['wrap_style'], center='com')

        # Write the coordinates

        # Output the xyz coordinates if we are making xys for AP
        if ts.frame > 0:
            if keywords["write_partitions"] and\
                    ts.frame % keywords["write_freq"] == 0:
                write_partitions(groups, ts, elements, indi.x_i, keywords)

        # End of loop
    return


def get_group_geom_center(u, types, ind, redundant=True, return_types=False):
    """
    Get the center of geometry for a group with indices given by the
    ind variable


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
    return_types: bool
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
    com = []
    ret_types = []
    num_types = len(types)
    type_str =  '(type %s' % (types[0])
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
    Return the index of a COM group
    Based on get_group_geom_center

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
    type_str =  '(type %s' % (types[0])
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
                rdh = sels.h.positions[m,:] - d_pos[k,:]
                rda = sels.a.positions[j,:] - d_pos[k,:]
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
    if don_ind != indi.donor -1:
        print("Swapping donors in residue based on H-bond dists at step: ", u.trajectory.frame)
        print("Old Don: {0}   New Don: {1}".format(indi.donor, don_ind +1))
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
                rdh = sels.h.positions[m,:] - d_pos[k,:]
                rda = sels.a.positions[j,:] - sels.h.positions[m,:]
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


def ratio_topology_change(u, indi, sels, all_u, ts, keywords, groups):
    """
    Adds a proton hop from one donor to the next depending on the D-H bond
    lengths for different atoms in the system.

    This can be used to describe tautomerization proton transfer reactions such
    as that for the proton between the two oxygens in glutamic acid.

    The proton is transferred if:

    r_km / [r_km + r_jm] > 0.5 (the distance between the donor and it's proton
                                is greater than an acceptor and the proton)

    :param u:
    :param indi:
    :param sels:
    :param all_u:
    :param ts:
    :param keywords:
    :param groups:
    :return:
    """
    if keywords["ind_method"] == 2:
        sels.set_donors(u, indi.donor)
    num_d = len(sels.d.indices)
    if num_d == 1:
        return
    indi.hop = []
    for k in range(num_d):
        for j in range(num_d):
            if j == k:
                continue
            sels.set_proton(u, sels.d.indices[k] + 1)
            num_h = len(sels.h.indices)
            for m in range(num_h):
                rdh = sels.d.positions[k,:] - sels.h.positions[m,:]
                rdh = np.linalg.norm(rdh)
                rah = sels.d.positions[j,:] - sels.h.positions[m,:]
                rah = np.linalg.norm(rah)
                p = rdh / (rdh + rah)
                if p > 0.5:
                    indi.hop.append((m , j, p, k))
    sels.set_dah(u, indi.donor)
    return

def calc_indicator(u, all_u, indi, sels, keywords, groups, ts):
    """
    Stage and calculate the location of the indicator.
    Different things happen depending on the indicator method.
    :param u: universe without indicator
    :param all_u: universe with indicator
    :param indi: indicator class
    :param sels: selection class
    :param keywords: dict with keywords
    :param groups: groups class
    :param ts: integer for timestep
    :return:
    """
    # Set up atom selections for donor and acceptors
    sels.set_dah(u, indi.donor)

    # Print out what the hydrogens, acceptors, donors are
    if keywords["indicator_verbose"]:
        print("#**************** STEP %d" % ts.frame)
        print("Donor indicies in order: ", sels.d.indices)
        print("Acc indices: ", sels.a.indices)
        print("Hyd indices: ", sels.h.indices)
    # Calculate the indicator location
    if keywords["ind_method"] == 0:
        indi.calc_indicator(sels.d.positions[0], sels.a.positions,
                            sels.h.positions, sels.d.types[0],
                            sels.a.types)
    elif keywords["ind_method"] == 1:
        don_com = get_group_geom_center(u, list(indi.rxh.keys()), [sels.d.indices[0]])[0]
        acc_com, atypes = get_group_geom_center(u, list(indi.rxh.keys()), sels.a.indices, redundant=False, return_types=True)
        acc_com = np.asarray(acc_com)

        indi.calc_indicator(don_com, acc_com,
                            sels.h.positions, sels.d.types[0],
                            atypes)
    elif keywords["ind_method"] == 2:
         check_for_donor_switch(u, indi, sels)
         indi.calc_indicator(sels.d.positions[0], sels.a.positions,
                            sels.h.positions, sels.d.types[0],
                            sels.a.types)
    elif keywords["ind_method"] == 3:
        don_com = [calculate_weighted_coords(u, indi, sels)]
        acc_com = get_group_geom_center(u, list(indi.rxh.keys()), sels.a.indices)
        h_coords = []
        for i in range(sels.d.n_atoms):
            sels.set_proton(u, sels.d.indices[i] + 1)
            h_coords.append(sels.h.positions)

        indi.calc_indicator(sels.d.positions, sels.a.positions,
                            h_coords, sels.d.types,
                            sels.a.types, don_com, acc_com)
    elif keywords["ind_method"] in [4, 6, 7, 9, 11]:
        don_com = get_group_geom_center(u, list(indi.rxh.keys()), [sels.d.indices[0]])
        acc_com = get_group_geom_center(u, list(indi.rxh.keys()), sels.a.indices)
        h_coords = []
        for i in range(sels.d.n_atoms):
            sels.set_proton(u, sels.d.indices[i] + 1)
            h_coords.append(sels.h.positions)

        indi.calc_indicator(sels.d.positions, sels.a.positions,
                            h_coords, sels.d.types,
                            sels.a.types, don_com, acc_com)

    elif keywords["ind_method"] in [8]:
        don_com = [calculate_weighted_coords(u, indi, sels)]
        acc_com = get_group_geom_center(u, list(indi.rxh.keys()), sels.a.indices)
        acc_com = np.asarray(acc_com)
        indi.calc_indicator(don_com, acc_com,
                            sels.h.positions, sels.d.types[0],
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
    sels = []
    for i in indi.correction_groups:
        selstr = 'bynum %d' % i[0]
        for j in i[1:]:
            selstr += ' or bynum %d' %j
        sels.append(u.select_atoms(selstr).positions)
    return sels

def do_hop(u, all_u, indi, ts, keywords, sels, groups, intra=False):
    """
    This subroutine has a lot going on
    We hop the proton, change the topology, and initialize the two universes
    based on the new topologies.
    :param u:
    :param all_u:
    :param indi:
    :param translation_vector:
    :param ts:
    :param keywords:
    :param sels:
    :return:
    """
    #
    # Set up the new indicies based on the indicator calculation
    #
    frame = ts.frame
    if len(indi.hop) > 1:
        print("Warning: multiple hops at frame ", frame)
        print("Choosing largest pmj")
        for i in range(1, len(indi.hop)):
            if indi.hop[i][2] > indi.hop[0][2]:
                indi.hop[0] = indi.hop[i]
    if keywords["ind_method"] == 7:
        if indi.hop[0][4] == True:
            intra = True
    if intra:
        if keywords["ind_method"] == 2:
            sels.set_donors(u, indi.donor)
        new_don = sels.d.indices[indi.hop[0][1]]
    else:
        new_don = sels.a.indices[indi.hop[0][1]]
    if keywords["ind_method"] in [3, 4, 6, 7, 9, 11]:
        old_don = sels.d.indices[indi.hop[0][3]]
    elif keywords["ind_method"] in [1]:
        inds = get_group_ind(u, list(indi.rxh.keys()), sels.a.indices, redundant=False)
        new_don = inds[indi.hop[0][1]]
        old_don = indi.donor - 1
    else:
        old_don = indi.donor - 1
    sels.set_proton(u, old_don + 1)
    transfer_h = sels.h.indices[indi.hop[0][0]]

    if keywords["verbose"]:
        print("Proton hop at frame ", ts.frame, "Donor: ", old_don + 1, "ACC:",
              new_don + 1)

    indi.donor = new_don + 1
    #
    # Change groups of atoms
    groups.transferAtom(transfer_h, new_don)
    #
    # Go through and get the new list of bonds for rewriting the topology
    new_bonds = calc_new_bonds_from_u(u, old_don, transfer_h, new_don)
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
                                *keywords["dimensions"], frame=frame)
    except:
        sys.exit("Error reloading universe after proton hop")
    if keywords['mcec']:
        new_all_u = add_indicator_to_universe(new_u, frame=frame, natoms=2)
    else:
        new_all_u = add_indicator_to_universe(new_u, frame=frame)
    sels.reset_all(new_u, new_all_u, indi.donor)
    indi.reset_hop()

    return new_u, new_all_u

def calc_new_bonds_from_u(u, old_don, transfer_h, new_don):
    """
    Here we make the assumption that the transfer H is only
    bonded to one atom. We just switch that bond
    This is to help with Method 1 PT
    :param u:
    :param old_don:
    :param transfer_h:
    :param new_don:
    :return:
    """
    num_bonds = u.bonds.indices.shape[0]
    new_bonds = []
    found_bond = False
    for b in range(num_bonds):
        # if old_don in u.bonds.indices[b, :2] and \
        #         transfer_h in u.bonds.indices[b, :2]:
        if transfer_h in u.bonds.indices[b,:2]:
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

def add_indicator_to_universe(u, ind_type="IND", ind_name="IND", frame=0,
                              natoms=1):
    """
    Add another atom and return the current universe
    :param u: universe with original coordinates
    :type MDAnalysis.Universe:
    :param ind_type: the new atom type for the indicator
    :type ind_type: str
    :param ind_name: the new atom name for the indicator
    :type ind_name: str
    :return: the new universe
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


def write_partitions(groups, ts, elements, center, keywords):
    from itertools import combinations
    """
    Write out the partitions that would be caculated if this were an adaptive
    partitioning calculation
    :param groups:
    :type groups: mdtools.Groups
    :param u: MDAnalysis universe without indicator
    :param center: center of active zone
    :return:
    """

    # Set up the variables
    r_a = keywords["active_radius"]
    r_b = keywords["buffer_radius"]
    r_ap = r_a + r_b

    # Get the group locations from the universe object
    group_positions = ts.positions[groups.groupReps]

    # Calculate the pairwise distances between the atoms and the center
    cen = center.reshape(1,3)
    dists = cdist(group_positions, cen).flatten()

    # Set up the ap zones
    active_groups = np.argwhere(dists < r_a)
    buffer_groups = np.argwhere((dists >= r_a) & (dists < r_ap) )


    # Calculate the partitions
    partition_groups = [[*active_groups]]
    for order in range(1, keywords["pap_order"] + 1):
        group_combos = list(combinations(buffer_groups, order))
        for combo in group_combos:
            partition_groups.append([*active_groups, *combo])

    # Calculate the atom indicies in the partitions
    partition_inds = []
    for i, p in enumerate(partition_groups):
        partition_inds.append([])
        for g in p:
            partition_inds[i].extend(groups.groupAtoms[np.asscalar(g)])

    # Write the partitions
    for i, p in enumerate(partition_inds):
        # generate the output name
        of_path = keywords["write_folder"]
        if keywords["write_prefix"] != "":
             of_path += keywords["write_prefix"] + '-'
        of_path += "{0:06d}_{1:03d}.".format(ts.frame, i)
        # write the file
        if keywords["write_type"] == "xyz":
            of_path += "xyz"
            ofi = open(of_path, 'w')
            mdtools.print_xyz(len(p), ts.positions[p,0],
                              ts.positions[p,1], ts.positions[p,2], elements[p],
                              ofi=ofi)
            ofi.close()
        else:
            print("Unsupported output type")
            sys.exit()


def initialize_universe(struct, coords, xdim, ydim, zdim, frame=0):
    """
    Initialize an m
    :param u: mdanalysis universe object
    :param ts: integer for timestep to jump to
    :param xdim: x dimension in A
    :param ydim: y dimension in A
    :param zdim: z dimension in A
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
