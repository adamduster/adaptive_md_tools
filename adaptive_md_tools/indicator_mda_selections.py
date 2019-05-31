#!/usr/bin/env python
"""
This module contains classes for selecting donors, acceptors, and transferring
protons for the indicator classes in indicator.py
"""
import warnings


class Selections:
    """
    This is the base selection class. It contains functionality for the
    mCEC, original indicator, and the new best indicator method (indicator B in
    the paper)
    """
    def __init__(self, u, all_u, proton_types, acceptor_types, rlist,
                 donor_index=None):
        """

        Parameters
        ----------
        u : MDAnalysis.universe object
            The MDA universe object without any proton indicator in it
        all_u: MDAnalysis.universe object
            The MDA universe object with the proton indicator appended to the
            end
        proton_types: list of str
            A list of atom types corresponding to the protons that qualify for
            the indicator calculations. Protons not in this list that are
            connected to the donor residue will be ignored
        acceptor_types: list of str
            A list of atom types corresponding to the acceptors that qualify for
            the indicator calculations.
        rlist: float
            Search radius for looking for acceptor molecules
        donor_index:
            1-based index for setting the initial donor of the selection.
            If this is present, the donor selection will automatically be
            set up. Otherwise, it is necessary to manually call another
            subroutine to select it

        Returns
        -------

        """
        #initialize acceptor selection string
        self.acc_string = "type %s" % acceptor_types[0]
        for acc in range(1, len(acceptor_types)):
            self.acc_string += " or type %s" % acceptor_types[acc]

        # initialize proton selection string
        self.proton_string = "type %s" % proton_types[0]
        for i in range(1, len(proton_types)):
            self.proton_string += " or type %s" % proton_types[i]

        # selection for donor
        self.d = None
        if donor_index:
            self.set_donor(u, donor_index)

        # selection for acceptor
        self.a = None

        # selection for hydrogen
        self.h = None

        # rlist parameter
        self.rlist = rlist
        # selection for system without indicator
        self.sys = None
        # selection for system with indicator
        self.all = None
        # Initialize selections
        self.reset_universe_selections(u, all_u)

    def set_donor(self, u, donor_ind):
        """
        Set the new donor(s) from a 1-based donor atom index.

        This will set the donor selection to a list of atoms that have:
            1. the same *resnum* as the input parameter donor_ind
            2. atom types from the self.acc_string variable

        Parameters
        ----------
        u : MDAnalysis.universe
            MDA universe to search for atoms within
        donor_ind: int
            1-based integer to base donor search from

        Returns
        -------

        """
        self.d = u.select_atoms("same resnum as bynum %d" % donor_ind)
        self.d = self.d.select_atoms(self.acc_string)
        if not self.d:
            print("Error selecting donor. Are the atom types listed correctly in the input file?")
            print("Selection string:")
            print("same resnum as bynum %d" % donor_ind)
            raise KeyError

    def set_acc(self, u):
        """
        Use the MDAnalysis selection tools to find the acceptors within
        self.rlist of the donor selection.

        Note:
            1. this selects atoms around all donor atoms in the self.d selection
            2. this deselects atoms that have the same resnum as the donor

        Parameters
        ----------
        u : MDAnalysis.universe
            MDA universe to search for atoms within
        """
        if not self.d:
            print("Error can't select acceptors because there is no donor")
            raise LookupError

        d_res = self.d.resids[0]
        sel_str = ("(({2}) and (around {0} ("
            + "bynum {1})))").format(self.rlist, self.d.indices[0]+1, self.acc_string)
        num_d = self.d.indices.size
        for d in range(1, num_d):
            sel_str += " or (({2}) and (around {0} (" \
                "bynum {1})))".format(self.rlist, self.d.indices[d]+1, self.acc_string)
        self.a = u.select_atoms(sel_str)
        self.a = self.a.select_atoms("not resnum %d " % d_res)

    def set_original_acc(self, u):
        """
        Uses the MDAnalysis selection tools to select acceptors within
        self.rlist of the list of self.d

        This does not deselect atoms that are in the same residue as the donor.

        It is called set_original_acc as it follows the rules of the original
        proton indicator.

        Parameters
        ----------
        u : MDAnalysis.universe
            MDA universe to search for atoms within
        """
        sel_str = ("(({2}) and (around {0} ("
            + "bynum {1})))").format(self.rlist, self.d.indices[0]+1, self.acc_string)
        num_d = self.d.indices.size
        for d in range(1, num_d):
            sel_str += " or (({2}) and (around {0} (" \
                "bynum {1})))".format(self.rlist, self.d.indices[d]+1, self.acc_string)
        self.a = u.select_atoms(sel_str)
        self.a = self.a.select_atoms("not bynum {0} ".format(self.d.indices[0]+1))

    def set_proton(self, u, atom_ind):
        """
        Select protons bonded to the atom given by atom_ind

        Parameters
        ----------
        u : MDAnalysis.universe
            MDA universe to search for atoms within
        atom_ind: int
            1-based integer to base donor search from
        """
        self.h = u.select_atoms(
            "({1}) and bonded (bynum {0})".format(atom_ind, self.proton_string))

    def set_dah(self, u, donor_ind):
        """
        Convenience function for setting the donor, then calculating the
        acceptor and proton selections.

        Parameters
        ----------
        u : MDAnalysis.universe
            MDA universe to search for atoms within
        donor_ind: int
            1-based integer to base donor search from
        """
        self.set_donor(u, donor_ind)
        self.set_acc(u, donor_ind)
        self.set_proton(u, donor_ind)

    def reset_universe_selections(self, u, all_u):
        """
        Selects all of the atoms in the universe and universe with indicator.

        Parameters
        ----------
        u : MDAnalysis.universe object
            The MDA universe object without any proton indicator in it
        all_u: MDAnalysis.universe object
            The MDA universe object with the proton indicator appended to the
            end
        """
        self.sys = u.select_atoms("all")
        self.all = all_u.select_atoms("all")

    def reset_all(self, u, all_u, donor_ind):
        """
        Convenience function for resetting the universe and indicator
        selections. Calls the reset universe function and then sets the donor,
        acceptor, and protons.

        Parameters
        ----------
        u : MDAnalysis.universe object
            The MDA universe object without any proton indicator in it
        all_u: MDAnalysis.universe object
            The MDA universe object with the proton indicator appended to the
            end
        donor_ind: int
            1-based integer to base donor search from
        """
        self.set_dah(u, donor_ind)
        self.reset_universe_selections(u, all_u)

    def set_one_donor(self, u, donor_ind):
        """
        Set the donor to be an individual atom.

        This is in contrast to the set_donor function which selects multiple
        atoms with the same resnum.

        Parameters
        ----------
        u : MDAnalysis.universe
            MDA universe to search for atoms within
        donor_ind: int
            1-based integer to base donor search from
        """
        self.d = u.select_atoms("bynum {0}".format(donor_ind))

    def set_all_acc(self, u):
        """
        Select all of the donor/acceptor atoms in the universe.
        This is used in conjunction with mCEC calculations.

        Parameters
        ----------
        u : MDAnalysis.universe
            MDA universe to search for atoms within
        """
        return u.select_atoms(self.acc_string)

    def set_all_protons(self, u):
        """
        Select all of the protons in the universe.
        This is used in conjunction with mCEC calculations.

        Parameters
        ----------
        u : MDAnalysis.universe
            MDA universe to search for atoms within
        """
        return u.select_atoms(self.proton_string)


class SelectionsInd1(Selections):
    """
    This is the selection class for indicator A in the paper.

    Here, we need to select all protons in the donor group. In
    the program, we just pretend the COM is the donor position
    and use this for the H positions.


    """
    def __init__(self, u, all_u, proton_types, acceptor_types, rlist, donor_index=None, a=None):
        """
        Initialize the selection class

        Parameters
        ----------
        u : MDAnalysis.universe object
            The MDA universe object without any proton indicator in it
        all_u: MDAnalysis.universe object
            The MDA universe object with the proton indicator appended to the
            end
        proton_types: list of str
            A list of atom types corresponding to the protons that qualify for
            the indicator calculations. Protons not in this list that are
            connected to the donor residue will be ignored
        acceptor_types: list of str
            A list of atom types corresponding to the acceptors that qualify for
            the indicator calculations.
        rlist: float
            Search radius for looking for acceptor molecules
        donor_index:
            1-based index for setting the initial donor of the selection.
            If this is present, the donor selection will automatically be
            set up. Otherwise, it is necessary to manually call another
            subroutine to select it

        """
        Selections.__init__(self, u, all_u, proton_types, acceptor_types, rlist, donor_index=donor_index, a=a)

    def set_proton(self, u, donor_ind):
        """
        Set all of the protons bonded do the donor group.

        Notes
        -----
        The below code suffers from some bug in glu-w where it selects properly
        when  '( bonded bynum 7 or bonded bynum 5 )'
        but not
             '( bonded bynum 5 or bonded bynum 7 )'

        I think I just had to switch the order in the input file...

        It can also run into a bug if two donors are bonded to proton

        Parameters
        ----------
        u : MDAnalysis.universe
            MDA universe to search for atoms within
        donor_ind: int
            1-based integer to base donor search from

        Returns
        -------

        """
        dinds = self.d.indices + 1
        self.h = u.select_atoms('bonded bynum {0}'.format(dinds[0]))
        for i in range(1, len(dinds)):
            self.h += u.select_atoms('bonded bynum {0}'.format(dinds[i]))
        self.h = self.h.select_atoms("{0}".format(self.proton_string))


class SelectionsInd2(Selections):
    """
    The selection class for indicator 2.

    This class is experimental and may not work as expected due to the fact that
    there was no congnizent equation for indicator 2.
    """
    def __init__(self, u, all_u, proton_types, acceptor_types, rlist, donor_index=None, a=None):
        Selections.__init__(self, u, all_u, proton_types, acceptor_types, rlist, donor_index=donor_index, a=a)
        warnings.warn("SelectionInd2 class is not implemented well")
        raise NotImplementedError

    def set_donors(self, u, donor_ind):
        self.d = u.select_atoms("same resnum as bynum %d" % donor_ind)
        self.d = self.d.select_atoms(self.acc_string)

    def set_donor(self, u, donor_ind):
        self.d = u.select_atoms("bynum %d" % donor_ind)
