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

from scipy.spatial.distance import cdist
import sys
import adaptive_md_tools.mdtools as mdtools
import numpy as np
import os


def write_sispa(groups, ts, elements, center, keywords):

    # Get the group locations from the universe object
    group_positions = ts.positions[groups.groupReps]

    active_groups, buffer_groups = find_active_buffer_groups(
        keywords["active_radius"], keywords["buffer_radius"], center,
        group_positions)

    # Calculate the Pi values
    buffer_dists = cdist(group_positions[buffer_groups], center.reshape(1, 3)).flatten()
    buffer_dists -= keywords["active_radius"]
    x = buffer_dists / keywords["buffer_radius"]

    buffer_pis = 1 - 10 * x ** 3 + 15 * x ** 4 - 6 * x ** 5

    # Make the coordinate and pi value lists
    # Lookup the active zone atomic indices and add 1 to the pi list for each
    # atom
    partition_inds = []  # Atom indices
    for group in active_groups:
        partition_inds.extend(groups.groupAtoms[np.asscalar(group)])
    pis = [1.] * len(partition_inds)

    # Lookup the buffer zone atomic indices and find the corresponding pi
    # value for each atom
    for i, group in enumerate(buffer_groups):
        atom_inds = groups.groupAtoms[group]
        num_atoms = len(atom_inds)
        partition_inds.extend(atom_inds)
        pis.extend([buffer_pis[i]] * num_atoms)

    # Write the SISPA geometry
    # generate the output name
    prefix = ''
    if keywords["write_prefix"] != "":
        prefix = keywords["write_prefix"] + '-'
    of_path = prefix + "{0:06d}.xyz".format(ts.frame)
    of_path = os.path.join(keywords["sispa_folder"], of_path)

    # write the file
    with open(of_path, 'w') as ofi:
        mdtools.print_sispa(len(partition_inds),
                            ts.positions[partition_inds, 0],
                            ts.positions[partition_inds, 1],
                            ts.positions[partition_inds, 2],
                            pis,
                            elements[partition_inds],
                            ofi=ofi)


def write_partitions(groups, ts, elements, center, keywords):
    """
    Write out the PAP partitions that would be calculated if this were an
    adaptive partitioning calculation.

    Parameters
    ----------
    groups: mdtools.Groups
        A groups object containing the adaptive partitioning groups
    ts: MDAnalysis.trajectory.timestep
        The current timestep
    elements:
    center: np.ndarray of floats with shape (3)
        The center of the active zone
    keywords: dict
        Program parameters
    """
    from itertools import combinations

    # Get the group locations from the universe object
    group_positions = ts.positions[groups.groupReps]

    active_groups, buffer_groups = find_active_buffer_groups(
        keywords["active_radius"], keywords["buffer_radius"], center,
        group_positions)

    # Calculate the PAP partitions
    partition_groups = [[*active_groups]]
    for order in range(1, keywords["pap_order"] + 1):
        group_combos = list(combinations(buffer_groups, order))
        for combo in group_combos:
            partition_groups.append([*active_groups, *combo])

    # Calculate the atom indices in the partitions
    partition_inds = []
    for i, p in enumerate(partition_groups):
        partition_inds.append([])
        for g in p:
            partition_inds[i].extend(groups.groupAtoms[np.asscalar(g)])

    # Write the partitions
    for i, p in enumerate(partition_inds):
        if keywords["ap_write_probability"]:
            if np.random.random() > keywords["ap_write_probability"]:
                continue
        # generate the output name
        of_path = keywords["write_folder"]
        if keywords["write_prefix"] != "":
            of_path += keywords["write_prefix"] + '-'
        of_path += "{0:06d}_{1:03d}.".format(ts.frame, i)
        # write the file
        if keywords["write_type"] == "xyz":
            of_path += "xyz"
            ofi = open(of_path, 'w')
            mdtools.print_xyz(len(p),
                              ts.positions[p, 0],
                              ts.positions[p, 1],
                              ts.positions[p, 2],
                              elements[p],
                              ofi=ofi)
            ofi.close()
        else:
            print("Unsupported output type")
            sys.exit()


def find_active_buffer_groups(active_radius, buffer_radius, center,
                              group_positions):
    """

    Parameters
    ----------
    active_radius: float
        radius from active center to edge of QM zone
    buffer_radius: float
        radius of buffer shell
    center: list of 3 floats
        x, y, and z coordinates of the center of the QM zone
    group_positions: np.ndarray of floats
        Nx3 array of coordinates representing the location of the representative
        atoms for the AP groups

    Returns
    -------
    active_groups: np.ndarray of int
        active group indices of which group_positions are in the active zone
    buffer_groups: np.ndarray of int
        buffer group indices of which group_positions are in the buffer zone
    """
    ap_radius = active_radius + buffer_radius

    # Calculate the pairwise distances between the atoms and the center
    cen = center.reshape(1, 3)
    dists = cdist(group_positions, cen).flatten()

    # Set up the ap zones
    active_groups = np.argwhere(dists < active_radius)
    buffer_groups = np.argwhere((dists >= active_radius) & (dists < ap_radius))

    return active_groups.flatten(), buffer_groups.flatten()
