#!/usr/bin/env python3
"""
This file contains classes for calculating the position of the indicator
with Numpy
"""
import sys
import numpy as np
from itertools import permutations
from numba import jit


class Indicator:
    """
    This is the implementation of the indicator as detailed in our 2019
    paper on EcCLC
    """

    def __init__(self):
        # Dictionary (atom_type: rdh) These are used to calculate
        # rdh0 and pmax
        self.rxh = {'OT': 1.0, 'SOT': 1.0, 'CLA': 1.4, 'OC': 1.0}
        # self.num_rdh0 = 0
        self.rlist = 3.5
        # The coordinates of the donor
        self.x_d = np.zeros(3, dtype=float)
        # Number of acceptors within rlist of donor
        self.num_acceptors = 0
        # Number of protons bound to donor
        self.num_h = 0
        # Location of the indicator
        self.x_i = np.zeros(3, dtype=float)
        # Identity of the donor
        self.donor = 0
        # Describes a proton hop
        # elements are: [[int:ind of proton, which acceptor hop to, pmj], ...]
        self.hop = []
        # Location of the acceptors
        self.x_as = []
        # Print flag
        self.print_all = False
        # Output reaction coordinates?
        self.output_freq = 0
        # Where to print indicator reaction coordinate information?
        self.log_path = 'indicator.log'
        self.xyz_path = 'donor.xyz'
        # Where to print debug information?
        self.ofi = sys.stdout
        # Log file object
        self._lfi = None
        self._xyz = None
        self.max_xyz_atoms = 5
        # This step
        self.step = 0

    def initialize_outind(self, path):
        "Open the outind file"
        self.indfi = open(path, 'w')
        print("In indicator outfile, first atom is indicator 2nd mcec")

    def set_output_freq(self, freq, prefix=''):
        """
        Initialize variables for writing the xyz and the log file

        Parameters
        ----------
        freq: int
            output frequency
        prefix: str
            file prefix
        """
        try:
            int(freq)
        except TypeError:
            sys.exit("Error: indicator output frequency must be an integer")
        if freq > 0:
            self.output_freq = freq
            if prefix != '':
                self.log_path = prefix + '-' + self.log_path
                self.xyz_path = prefix + '-' + self.xyz_path
            self._lfi = open(self.log_path, 'w')
            self._lfi.write("# Step           rho             dr\n")
            self._xyz = open(self.xyz_path, 'w')

    def reset_hop(self):
        """
        After proton hop, clear the list of hops
        """
        self.hop = []

    def _write_log(self, p, dr, coords=None):
        """
        Write the results of a step
        Write the rho and dr coordinates
        If a matrix of coordinates is present, write them to the xyz file

        Parameters
        ----------
        :param p:
        :param dr:
        :param coords:
        :return:
        """
        self._lfi.write('{0:10d}  {1:10.6f}   {2:10.6f}\n'.format(
            self.step, p, dr))

        if coords is None:
            return
        #if not coords.any():
        #    return

        natoms = coords.shape[0]

        try:
            self._xyz.write('%d\n' % self.max_xyz_atoms)
        except:
            sys.exit("Error writing number of atoms")
        self._xyz.write('\n')

        xyz_str = "{3}     {0:12.6}     {1:12.6f}     {2:12.6f}\n"
        els = ["Ti", "V ", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge"]
        try:
            for i in range(natoms):
                self._xyz.write(xyz_str.format(*coords[i], els[i]))
            for j in range(i + 1, self.max_xyz_atoms):
                self._xyz.write(xyz_str.format(0., 0., 0., els[j]))
        except IOError:
            print("Error writing coords")
            raise
        return

    def add_rdh0(self, rdh, atom_type: str):
        """
        Add a rho parameter and atom type to the dict of parameters

        Parameters
        ----------
        rdh: float
            equilibrium dh distance in angstrom
        atom_type: str
            atom type to associate with rdh value

        """
        try:
            if rdh < 0:
                print("Error, rdh must be greater than 0.")
                raise ValueError('add_rdh 1')
        except TypeError('add_rdh 2'):
            print("Error, excpected a number for rdh ")
            sys.exit()

        try:
            atom_upper = atom_type.upper()
        except TypeError('add_rdh3'):
            print("Error parsing atom type {0}".format(atom_type))
            sys.exit()

        try:
            self.rxh[atom_type] = rdh
        except TypeError("add rdh 4"):
            print("Error adding atom {0} and rdh parameter {1} to "
                  "rdh".format(atom_type, rdh))
        return

    def print_rdh0(self, ofi=None):
        """
        Output data for double checking

        Parameters
        ----------
        ofi: file object with write permissions
            The output file object. If None, then this will print to stdout
        """
        print("Atom_Type  rDH", file=ofi)
        for key, rdh in self.rxh:
            print("{0:8}    {0:5.3f}".format(key, rdh), file=ofi)
        return

    def calc_indicator(self, x_d, x_as, x_hms, type_d, type_as, ofi=None):
        """
        This is the main subroutine for calculating the indicator

        Parameters
        ----------
        x_d: ndarray of float
            coordinates with shape [3]
        x_as: ndarray of float
            acceptor coordinates each with shape [j,3]
        x_hms: ndarray of float
            hydrogen coordinates each with shape [m,3]
        type_d: str
            donor type to link with rho parameters
        type_as: list of str
            acceptor types to link with rho parameters
        ofi: file object
            where to print stuff
        """
        cstr = "{0:0.5f}   {1:0.5f}   {2:0.5f}\n"
        icstr = "{0:9d}   {1:9.5f}   {2:9.5f}   {3:9.5f}\n"

        # Set donor coordinates
        try:
            self.x_d[:] = x_d[:]
        except RuntimeError("calc_indicator 1"):
            print("Error setting donor coordinates")
            sys.exit()

        # Set acceptor coordinates
        try:
            self.num_acceptors = x_as.shape[0]
            if self.num_acceptors == 0:
                print("Warning: NO ACCEPTOR")
                print("Setting indicator location to donor coordinates")
                self.x_i[:] = self.x_d[:]
                return 1
            self.x_as = np.zeros((self.num_acceptors, 3), dtype=float)
            self.x_as[:, :] = x_as[:, :]
        except RuntimeError("calc_indicator 2"):
            print("Error setting acceptor coordinates")
            sys.exit()

        # Set hydrogen coordinates
        self.num_h = x_hms.shape[0]
        if self.num_h <= 0:
            print("Error, no protons for indicator")
            raise RuntimeError("calc_indicator 4")
        try:
            self.x_hs = np.zeros((self.num_h, 3), dtype=float)
            self.x_hs = x_hms[:, :]
        except RuntimeError("Calc indicator 3"):
            print("Error setting Hydrogen coordinates")
            sys.exit()

        # Initialize the rho parameters
        try:
            rdh0 = self.rxh[type_d]
        except RuntimeError:
            print("Error hashing donor. Is donor in rdh0 list? is only one"
                  " donor passed to subroutine?")
            raise
        pmaxs = np.asarray([rdh0 / (rdh0 + self.rxh[a]) for a in type_as])
        pmj0 = rdh0 / self.rlist

        # Initialize the other arrays
        dims = (self.num_h, self.num_acceptors)
        pmjs = np.zeros(dims, dtype=float)
        xmjs = np.zeros(dims, dtype=float)
        gmjs = np.zeros(dims, dtype=float)
        self.hop = []

        #Begin the calculations
        largest_p = 0
        dr = 0
        for m in range(self.num_h):
            for j in range(self.num_acceptors):
                pmjs[m, j] = self.calc_pmj(self.x_d, self.x_as[j], self.x_hs[m])
                if pmjs[m, j] > pmaxs[j]:
                    self.hop.append((m, j, pmjs[m, j]))
                if pmjs[m, j] > largest_p:
                    largest_p = pmjs[m, j]
                    dr = np.linalg.norm(self.x_d - self.x_hs[m]) - \
                         np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                xmjs[m, j] = self.calc_xmj(pmjs[m, j], pmj0, pmaxs[j])
                gmjs[m, j] = self.calc_gmj(xmjs[m, j])

        gI = self.calc_gI(gmjs)
        self.x_i[:] = self.x_d[:]
        for j in range(self.num_acceptors):
            for m in range(self.num_h):
                self.x_i[:] += gmjs[m, j] * x_as[j]
        self.x_i *= 1. / gI

        if self.print_all:
            self.ofi.write("Detailed Stats\n")
            self.ofi.write("Donor Coords:\n")
            self.ofi.write(cstr.format(*self.x_d))
            self.ofi.write("Acceptor Coords:\n")
            for j in range(self.num_acceptors):
                self.ofi.write(icstr.format(j, *self.x_as[j]))
            self.ofi.write("Proton Coords\n")
            for m in range(self.num_h):
                self.ofi.write(icstr.format(m, *self.x_hs[m]))
            self.ofi.write("pmjs\n")
            print(pmjs, file=ofi)
            self.ofi.write("xmjs\n")
            print(xmjs, file=ofi)
            self.ofi.write("gmjs\n")
            print(gmjs, file=ofi)
            self.ofi.write("gI\n")
            print(gI, file=ofi)
            self.ofi.write("Hops:\n")
            print(self.hop, file=ofi)
        if self.output_freq:
            if self.step % self.output_freq == 0:
                self._write_log(largest_p, dr, coords=self.x_d.reshape(-1, 3))
        self.step += 1
        return 0

    @staticmethod
    @jit
    def calc_pmj(x_d, x_aj, x_hm):
        """
        calculate the variable p_mj [ rho_mj ]
        this is the projection of the D-H vec onto the D-A vec

        Parameters
        ----------
        x_d: ndarray of float with shape(3)
            coordinates of donor (np.float array [3] )
        x_aj: ndarray of float with shape(3)
            coordinates of acceptor j (np.float array [3] )
        x_hm: ndarray of float with shape(3)
            coordinates of hydrogen m (np.float array [3] )

        Returns
        -------
        float
        """
        r_dhm = x_hm - x_d
        r_daj = x_aj - x_d
        return np.dot(r_dhm, r_daj) / np.linalg.norm(r_daj)**2

    @staticmethod
    @jit(nopython=True)
    def calc_xmj(pmj, pmj0, pmax):
        """
        calculate the variable x(p_mj) [ x(rho_mj) ]. This is the ratio
        that deals with how far we are away from equilibrium.

        Parameters
        ----------
        pmj:  float
            projection scalar
        pmj0: float
            scaling parameter parameter
        pmax: float
            equilibrium bond constant ratio

        Returns
        -------
        x_pmj: float
        """
        return 1 - (pmj - pmj0) / (pmax - pmj0)

    @staticmethod
    @jit(nopython=True)
    def calc_gmj(xmj):

        if 1 <= xmj:
            gmj = 0.
        elif xmj < 0:
            gmj = 1
        else:
            gmj = -6 * xmj**5 + 15 * xmj**4 - 10 * xmj**3 + 1
        return gmj

    @staticmethod
    @jit
    def calc_gI(gmjs):
        """
        Calculate the normalization constant gI

        Parameters
        ----------
        :param gmjs: the splined projection vectors
        :type gmjs: np.ndarray
        :return: the normalization constant
        :rtype: np.float
        """
        return 1 + np.sum(gmjs)


# class Indicator2(Indicator):
#     """
#     This implementation of the indicator is the one where the projection
#     vectors between donors in a group and other acceptors are calcualted.
#     The results from each donor are then added back to calculate the final
#     location for the indicator.
#
#     X_I = 1./g_I * [ X_D_com + \sum_k \sum_j \sum_m { rho_kmj * X_A_j } ] s
#
#     """
#     def __init__(self):
#         Indicator.__init__(self)
#
#     def calc_indicator(self, x_d, x_as, x_hms, type_d, type_as, intrap, xk, ofi=None):
#         """
#         This is the main subroutine for calculating the indicator
#         :param x_d: ndarray float coordinates with shape [3]
#         :param x_as: ndarray of acceptor coordinates each with shape [j,3]
#         :param x_hms: ndarray of hydrogen coordiantes each with shape [m,3]
#         :param type_d: string of donor types to link with rho parameters
#         :param type_as: list of strings of acceptor types to link with rho parameters
#         :param ofi: where to print stuff
#         :return:
#         """
#         cstr = "{0:0.5f}   {1:0.5f}   {2:0.5f}\n"
#         icstr = "{0:9d}   {1:9.5f}   {2:9.5f}   {3:9.5f}\n"
#
#         # Set donor coordinates
#         try:
#             self.x_d[:] = x_d[:]
#         except RuntimeError("calc_indicator 1"):
#             print("Error setting donor coordinates")
#             sys.exit()
#
#         # Set acceptor coordinates
#         try:
#             self.num_acceptors = x_as.shape[0]
#             if self.num_acceptors == 0:
#                 print("WARNING: NO ACCEPTOR")
#                 return 1
#             self.x_as = np.zeros((self.num_acceptors, 3), dtype=float)
#             self.x_as[:, :] = x_as[:, :]
#         except RuntimeError("calc_indicator 2"):
#             print("Error setting acceptor coordinates")
#             sys.exit()
#
#         # Set hydrogen coordinates
#         self.num_h = x_hms.shape[0]
#         if self.num_h <= 0:
#             print("Error, no protons for indicator")
#             raise RuntimeError("calc_indicator 4")
#         try:
#             self.x_hs = np.zeros((self.num_h, 3), dtype=float)
#             self.x_hs = x_hms[:, :]
#         except RuntimeError("Calc indicator 3"):
#             print("Error setting Hydrogen coordinates")
#             sys.exit()
#
#         # Initialize the rho parameters
#         try:
#             rdh0 = self.rxh[type_d]
#         except RuntimeError:
#             "Error hashing donor. Is donor in rdh0 list? is only one donor " \
#             "passed to subroutine?"
#         pmaxs = np.asarray([rdh0 / (rdh0 + self.rxh[a]) for a in type_as])
#         pmj0 = rdh0 / self.rlist
#
#         # Initialize the other arrays
#         dims = (self.num_h, self.num_acceptors)
#         pmjs = np.zeros(dims, dtype=float)
#         xmjs = np.zeros(dims, dtype=float)
#         gmjs = np.zeros(dims, dtype=float)
#         self.hop = []
#
#         #Begin the calculations
#         largest_p = 0
#         dr = 0
#         for m in range(self.num_h):
#             for j in range(self.num_acceptors):
#                 pmjs[m,j] = self.calc_pmj(self.x_d, self.x_as[j], self.x_hs[m])
#                 if pmjs[m, j] > pmaxs[j]:
#                     self.hop.append((m, j, pmjs[m, j]))
#                 if pmjs[m, j] > largest_p:
#                     largest_p = pmjs[m, j]
#                     dr = np.linalg.norm(self.x_d - self.x_hs[m]) - \
#                          np.linalg.norm(self.x_as[j] - self.x_hs[m] )
#                 xmjs[m,j] = self.calc_xmj(pmjs[m,j], pmj0, pmaxs[j])
#                 gmjs[m,j] = self.calc_gmj(xmjs[m,j])
#
#         self.x_i[:] = self.x_d[:]
#
#         found_intra_hop = False
#         intrag = 0
#         for pp in range(len(intrap)):
#             if intrap[pp] > 0.5:
#                 if not found_intra_hop:
#                     self.hop = []
#                     found_intra_hop = True
#                 self.hop.append((m, j, intrap[pp], k, True))
#             xmk = self.calc_xmj( pmj0, pmaxs[j])
#             gmk = self.calc_gmj(xmjs[m, j])
#             my_g = self.calc_gmj(my_p_don)
#             self.x_i += my_g * self.x_d[k]
#             sum_gs += my_g
#
#         gI = self.calc_gI(gmjs)
#         for j in range(self.num_acceptors):
#             for m in range(self.num_h):
#                 self.x_i[:] += gmjs[m,j] * x_as[j]
#         self.x_i *= 1. / gI
#
#         if self.print_all:
#             self.ofi.write("Detailed Stats\n")
#             self.ofi.write("Donor Coords:\n")
#             for k in len(self.x_d):
#                 self.ofi.write(cstr.format(*self.x_d[k]))
#             self.ofi.write("Acceptor Coords:\n")
#             for j in range(self.num_acceptors):
#                 self.ofi.write(icstr.format(j, *self.x_as[j]))
#             self.ofi.write("Proton Coords\n")
#             for m in range(self.num_h):
#                 self.ofi.write(icstr.format(m, *self.x_hs[m]))
#             self.ofi.write("pmjs\n")
#             print(pmjs, file=ofi)
#             self.ofi.write("xmjs\n")
#             print(xmjs, file=ofi)
#             self.ofi.write("gmjs\n")
#             print(gmjs, file=ofi)
#             self.ofi.write("gI\n")
#             print(gI, file=ofi)
#             self.ofi.write("Hops:\n")
#             print(self.hop, file=ofi)
#         if self.output_freq:
#             if self.step % self.output_freq == 0:
#                 self._write_log(largest_p, dr, coords=self.x_d.reshape(-1, 3))
#         self.step += 1
#         return 0

class IndicatorNull(Indicator):
    """
    Indicator class if there is no indicator
    """

    def __init__(self):
        Indicator.__init__(self)


class Indicator4(Indicator):
    """
    This implementation of the indicator is the one where the projection
    vectors between donors in a group and other acceptors are calcualted.
    The results from each donor are then added back to calculate the final
    location for the indicator.

    X_I = 1./g_I * [ X_D_com + \sum_k \sum_j \sum_m { rho_kmj * X_A_j } ] s

    """

    def __init__(self):
        Indicator.__init__(self)
        self.donor_com = []
        self.acceptor_com = []

    def calc_indicator(self, x_d, x_as, x_hms, type_d, type_as, d_com, as_com,
                       ofi=None):
        """

        Parameters
        ----------
        :param x_d:  ndarray of donor coordinates with shape [k,3]
        :param x_as: ndarray of acc coordinates with shape [j,3]
        :param x_hms: ndarra of hyd coordinates with shape [m,3]
        :param type_d: list of strings with length k
        :param type_as: list of strings with length j
        :param d_com: list of length 1 with [3] array !TODO make this less wacky
        :param as_com: list of acceptor centers of mass
        :param ofi: where to print stuff
        :return:
        """
        cstr = "{0:0.5f}   {1:0.5f}   {2:0.5f}\n"
        icstr = "{0:9d}   {1:9.5f}   {2:9.5f}   {3:9.5f}\n"

        num_d = len(x_d)
        # Set donor coordinates
        try:
            self.x_d = x_d.copy()
            self.d_com = d_com.copy()[0]
        except RuntimeError("calc_indicator 1"):
            print("Error setting donor coordinates")
            sys.exit()

        # Set acceptor coordinates
        try:
            self.num_acceptors = x_as.shape[0]
            if self.num_acceptors == 0:
                print("Warning: NO ACCEPTOR")
                print("Setting indicator location to donor com")
                self.x_i[:] = self.d_com[:]
                return 1
            self.x_as = np.zeros((self.num_acceptors, 3), dtype=float)
            self.x_as[:, :] = x_as[:, :]
        except RuntimeError("calc_indicator 2"):
            print("Error setting acceptor coordinates")
            sys.exit()

        if self.print_all:
            self.ofi.write("*****STEP %d\n" % self.step)
            self.ofi.write("Donor Coords:\n")
            for k in range(len(x_d)):
                self.ofi.write(cstr.format(*self.x_d[k]))
            self.ofi.write("Acceptor Coords:\n")
            for j in range(self.num_acceptors):
                self.ofi.write(icstr.format(j, *self.x_as[j]))
            self.ofi.write("Proton Coords\n")
            self.ofi.write("Donor COM\n")
            print(d_com, file=ofi)
            print("Acceptor COMs", file=ofi)
            print(as_com)

        gI = 0
        self.x_i = np.zeros(3)
        largest_p = 0
        dr = 0
        self.hop = []
        sum_gs = 0.0
        found_proton = False

        for k in range(num_d):

            # Set hydrogen coordinates
            self.num_h = np.shape(x_hms[k])[0]
            if self.num_h <= 0:
                continue
            found_proton = True
            try:
                self.x_hs = np.zeros((self.num_h, 3), dtype=float)
                self.x_hs = x_hms[k][:, :]
            except RuntimeError("Calc indicator 3"):
                print("Error setting Hydrogen coordinates")
                sys.exit()

            # Initialize the rho parameters
            try:
                rdh0 = self.rxh[type_d[k]]
            except RuntimeError:
                print("Error hashing donor. Is donor in rdh0 list? is only"
                      " one donor passed to subroutine?")
                raise
            pmaxs = np.asarray([rdh0 / (rdh0 + self.rxh[a]) for a in type_as])
            pmj0 = rdh0 / self.rlist

            # Initialize the other arrays
            dims = (self.num_h, self.num_acceptors)
            pmjs = np.zeros(dims, dtype=float)
            xmjs = np.zeros(dims, dtype=float)
            gmjs = np.zeros(dims, dtype=float)

            #Begin the calculations
            for m in range(self.num_h):
                for j in range(self.num_acceptors):
                    pmjs[m, j] = self.calc_pmj(self.x_d[k], self.x_as[j],
                                               self.x_hs[m])
                    if pmjs[m, j] > pmaxs[j]:
                        self.hop.append((m, j, pmjs[m, j], k))
                    if pmjs[m, j] > largest_p:
                        largest_p = pmjs[m, j]
                        dr = np.linalg.norm(self.x_d[k] - self.x_hs[m]) - \
                             np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                    xmjs[m, j] = self.calc_xmj(pmjs[m, j], pmj0, pmaxs[j])
                    gmjs[m, j] = self.calc_gmj(xmjs[m, j])
                    print("Correction:  ", x_d[k] - d_com[0])
                    # Add the weighted Acceptor cog coordinate
                    self.x_i[:] += gmjs[m, j] * as_com[j]
            sum_gs += np.sum(gmjs[:])

            if self.print_all:
                print("For donor %d" % k)
                print("Hydrogen coordinates:")
                for m in range(self.num_h):
                    self.ofi.write(icstr.format(m, *self.x_hs[m]))
                self.ofi.write("pmjs\n")
                print(pmjs, file=ofi)
                self.ofi.write("xmjs\n")
                print(xmjs, file=ofi)
                self.ofi.write("gmjs\n")
                print(gmjs, file=ofi)
                self.ofi.write("pmaxs\n")
                print(pmaxs, file=ofi)
                self.ofi.write("Hops:\n")
                print(self.hop, file=ofi)
                print("cumulative_sum_of_gI", sum_gs, file=ofi)

        if not found_proton:
            print("Error, no protons for indicator")
            raise RuntimeError("calc_indicator 4")

        gI = self.calc_gI(sum_gs)

        self.calc_ind(d_com, gI)

        if self.print_all:
            self.ofi.write("gI\n")
            print(gI, file=ofi)
            print("Final location", file=ofi)
            print(self.x_i)

        if self.output_freq:
            if self.step % self.output_freq == 0:
                self._write_log(largest_p, dr, d_com[0][np.newaxis, :])
        self.step += 1
        return 0

    def calc_ind(self, d_com, gI):
        self.x_i += d_com[0][:]
        self.x_i /= gI


class Indicator6(Indicator4):
    """
    Modify indicator 4 by exponentiating the gmj terms

    g(pmj) -> exp[g(pmj)]
    gI -> e + sum{exp[g(pmj)]}
    """

    def __init__(self):
        Indicator4.__init__(self)

    def calc_gmj(self, xmj):
        if 1 <= xmj:
            gmj = 0.
        elif xmj < 0:
            gmj = 1
        else:
            gmj = -6 * xmj**5 + 15 * xmj**4 - 10 * xmj**3 + 1
        return gmj * np.exp(gmj)

    def calc_ind(self, d_com, gI):
        self.x_i += d_com[0][:] * np.e
        self.x_i /= gI

    @staticmethod
    @jit(nopython=True)
    def calc_gI(gmjs):
        """
        Calculate the normalization constant gI

        Parameters
        ----------
        :param gmjs: the splined projection vectors
        :type gmjs: np.ndarray
        :return: the normalization constant
        :rtype: np.float
        """
        return np.e + np.sum(gmjs)


class Indicator7(Indicator4):
    """
    This is indicator 4 with the intramolecular rho's added to the location
    """

    def __init__(self):
        Indicator4.__init__(self)

    def calc_indicator(self,
                       x_d,
                       x_as,
                       x_hms,
                       type_d,
                       type_as,
                       d_com,
                       as_com,
                       ofi=None):
        cstr = "{0:0.5f}   {1:0.5f}   {2:0.5f}\n"
        icstr = "{0:9d}   {1:9.5f}   {2:9.5f}   {3:9.5f}\n"

        num_d = len(x_d)
        self.num_d = num_d
        # Set donor coordinates
        try:
            self.x_d = x_d.copy()
        except RuntimeError("calc_indicator 1"):
            print("Error setting donor coordinates")
            sys.exit()

        # Set acceptor coordinates
        try:
            self.num_acceptors = x_as.shape[0]
            if self.num_acceptors == 0:
                print("WARNING: NO ACCEPTOR")
                return 1
            self.x_as = np.zeros((self.num_acceptors, 3), dtype=float)
            self.x_as[:, :] = x_as[:, :]
        except RuntimeError("calc_indicator 2"):
            print("Error setting acceptor coordinates")
            sys.exit()

        if self.print_all:
            self.ofi.write("*****STEP %d\n" % self.step)
            self.ofi.write("Donor Coords:\n")
            for k in range(len(x_d)):
                self.ofi.write(cstr.format(*self.x_d[k]))
            self.ofi.write("Acceptor Coords:\n")
            for j in range(self.num_acceptors):
                self.ofi.write(icstr.format(j, *self.x_as[j]))
            self.ofi.write("Proton Coords\n")
            self.ofi.write("Donor COM\n")
            print(d_com, file=ofi)
            print("Acceptor COMs", file=ofi)
            print(as_com)

        gI = 0
        self.x_i = np.zeros(3)
        largest_p = 0
        dr = 0
        self.hop = []
        sum_gs = 0.0
        intra_p = False
        p_don = 0

        # Calculate the intramolecular rhos
        for k in range(num_d):

            # Set hydrogen coordinates
            self.num_h = x_hms[k].shape[0]
            # if self.num_h <= 0:
            #     print("Error, no protons for indicator")
            #     raise RuntimeError("calc_indicator 4")
            try:
                self.x_hs = np.zeros((self.num_h, 3), dtype=float)
                self.x_hs = x_hms[k][:, :]
            except RuntimeError("Calc indicator 3"):
                print("Error setting Hydrogen coordinates")
                sys.exit()

            # Initialize the rho parameters
            try:
                rdh0 = self.rxh[type_d[k]]
            except RuntimeError:
                print(
                "Error hashing donor. Is donor in rdh0 list? is only one donor "
                "passed to subroutine?")
                raise
            pmaxs = np.asarray([rdh0 / (rdh0 + self.rxh[a]) for a in type_as])
            pmj0 = rdh0 / self.rlist

            # Initialize the other arrays
            dims = (self.num_h, self.num_acceptors)
            pmjs = np.zeros(dims, dtype=float)
            xmjs = np.zeros(dims, dtype=float)
            gmjs = np.zeros(dims, dtype=float)

            #Begin the calculations
            for m in range(self.num_h):
                for j in range(self.num_acceptors):
                    pmjs[m, j] = self.calc_pmj(self.x_d[k], self.x_as[j],
                                               self.x_hs[m])
                    if pmjs[m, j] > pmaxs[j] and not intra_p:
                        self.hop.append((m, j, pmjs[m, j], k, False))
                    if pmjs[m, j] > largest_p:
                        largest_p = pmjs[m, j]
                        dr = np.linalg.norm(self.x_d[k] - self.x_hs[m]) - \
                             np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                    xmjs[m, j] = self.calc_xmj(pmjs[m, j], pmj0, pmaxs[j])
                    gmjs[m, j] = self.calc_gmj(xmjs[m, j])
                    # self.x_i[:] += gmjs[m, j] * (2*x_as[j] - as_com[j] + d_com[0] - x_d[k])
                    # self.x_i[:] += gmjs[m, j] * (x_as[j])
                    self.x_i[:] += gmjs[m, j] * (as_com[j])
            sum_gs += np.sum(gmjs[:])

            if self.print_all:
                print("For donor %d" % k)
                for m in range(self.num_h):
                    self.ofi.write(icstr.format(m, *self.x_hs[m]))
                self.ofi.write("pmjs\n")
                print(pmjs, file=ofi)
                self.ofi.write("xmjs\n")
                print(xmjs, file=ofi)
                self.ofi.write("gmjs\n")
                print(gmjs, file=ofi)
                self.ofi.write("pmaxs\n")
                print(pmaxs, file=ofi)
                self.ofi.write("Hops:\n")
                print(self.hop, file=ofi)
                print("cumulative_sum_of_gI", sum_gs, file=ofi)

        found_intra_hop = False
        for k in range(num_d):
            self.num_h = x_hms[k].shape[0]
            try:
                self.x_hs = np.zeros((self.num_h, 3), dtype=float)
                self.x_hs = x_hms[k][:, :]
            except RuntimeError("Calc indicator 9"):
                print("Error setting Hydrogen coordinates")
                sys.exit()
            for j in range(self.num_d):
                if j == k:
                    continue
                for m in range(self.num_h):
                    rah = np.linalg.norm(self.x_hs[m] - self.x_d[j])
                    my_p_don = rah / (
                        rah + np.linalg.norm(self.x_hs[m] - self.x_d[k]))
                    if my_p_don < 0.5:
                        if not found_intra_hop:
                            self.hop = []
                            found_intra_hop = True
                        self.hop.append((m, j, my_p_don, k, True))
                    my_g = self.calc_gmj(my_p_don)
                    self.x_i += my_g * self.x_d[k]
                    sum_gs += my_g

        gI = self.calc_gI(sum_gs)

        self.calc_ind(d_com, gI)

        if self.print_all:
            self.ofi.write("gI\n")
            print(gI, file=ofi)
            print("Final location", file=ofi)
            print(self.x_i)

        if self.output_freq:
            if self.step % self.output_freq == 0:
                self._write_log(largest_p, dr)
        self.step += 1

        return 0


class Indicator9(Indicator4):
    """
    Indicator 4 with weighting of the donor coordinate by the following formula

    X_k_w = [r_km - rDH^0] / [\sum_k (\sum_m_k r_km - rDH^0)]

    It worked slightly better in some cases than 4 but was
    very sensitive to vibrations of molecular bonds at equilbirium
    """

    def __init__(self):
        Indicator4.__init__(self)

    def calc_indicator(self,
                       x_d,
                       x_as,
                       x_hms,
                       type_d,
                       type_as,
                       d_com,
                       as_com,
                       ofi=None):
        cstr = "{0:0.5f}   {1:0.5f}   {2:0.5f}\n"
        icstr = "{0:9d}   {1:9.5f}   {2:9.5f}   {3:9.5f}\n"

        num_d = len(x_d)
        # Set donor coordinates
        try:
            self.x_d = x_d.copy()
            self.d_com = d_com.copy()[0]
        except RuntimeError("calc_indicator 1"):
            print("Error setting donor coordinates")
            sys.exit()

        # Set acceptor coordinates
        try:
            self.num_acceptors = x_as.shape[0]
            if self.num_acceptors == 0:
                print("WARNING: NO ACCEPTOR")
                return 1
            self.x_as = np.zeros((self.num_acceptors, 3), dtype=float)
            self.x_as[:, :] = x_as[:, :]
        except RuntimeError("calc_indicator 2"):
            print("Error setting acceptor coordinates")
            sys.exit()

        if self.print_all:
            self.ofi.write("*****STEP %d\n" % self.step)
            self.ofi.write("Donor Coords:\n")
            for k in range(len(x_d)):
                self.ofi.write(cstr.format(*self.x_d[k]))
            self.ofi.write("Acceptor Coords:\n")
            for j in range(self.num_acceptors):
                self.ofi.write(icstr.format(j, *self.x_as[j]))
            self.ofi.write("Proton Coords\n")
            self.ofi.write("Donor COM\n")
            print(d_com, file=ofi)
            print("Acceptor COMs", file=ofi)
            print(as_com)

        gI = 0
        self.x_i = np.zeros(3)
        largest_p = 0
        dr = 0
        self.hop = []
        sum_gs = 0.0
        found_proton = False

        # Calculate the new weighting
        my_dcom = d_com
        if num_d > 1:
            my_dcom = self.calc_d_weights(type_d, x_hms)

        for k in range(num_d):

            # Set hydrogen coordinates
            self.num_h = x_hms[k].shape[0]
            if self.num_h <= 0:
                continue
            found_proton = True
            try:
                self.x_hs = np.zeros((self.num_h, 3), dtype=float)
                self.x_hs = x_hms[k][:, :]
            except RuntimeError("Calc indicator 3"):
                print("Error setting Hydrogen coordinates")
                sys.exit()

            # Initialize the rho parameters
            try:
                rdh0 = self.rxh[type_d[k]]
            except RuntimeError:
                "Error hashing donor. Is donor in rdh0 list? is only one donor " \
                "passed to subroutine?"
            pmaxs = np.asarray([rdh0 / (rdh0 + self.rxh[a]) for a in type_as])
            pmj0 = rdh0 / self.rlist

            # Initialize the other arrays
            dims = (self.num_h, self.num_acceptors)
            pmjs = np.zeros(dims, dtype=float)
            xmjs = np.zeros(dims, dtype=float)
            gmjs = np.zeros(dims, dtype=float)

            #Begin the calculations
            for m in range(self.num_h):
                for j in range(self.num_acceptors):
                    pmjs[m, j] = self.calc_pmj(self.x_d[k], self.x_as[j],
                                               self.x_hs[m])
                    if pmjs[m, j] > pmaxs[j]:
                        self.hop.append((m, j, pmjs[m, j], k))
                    if pmjs[m, j] > largest_p:
                        largest_p = pmjs[m, j]
                        dr = np.linalg.norm(self.x_d[k] - self.x_hs[m]) - \
                             np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                    xmjs[m, j] = self.calc_xmj(pmjs[m, j], pmj0, pmaxs[j])
                    gmjs[m, j] = self.calc_gmj(xmjs[m, j])
                    # self.x_i[:] += gmjs[m, j] * (2*x_as[j] - as_com[j] + d_com[0] - x_d[k])
                    # self.x_i[:] += gmjs[m, j] * (x_as[j])
                    self.x_i[:] += gmjs[m, j] * (as_com[j])
            sum_gs += np.sum(gmjs[:])

            if self.print_all:
                print("For donor %d" % k)
                for m in range(self.num_h):
                    self.ofi.write(icstr.format(m, *self.x_hs[m]))
                self.ofi.write("pmjs\n")
                print(pmjs, file=ofi)
                self.ofi.write("xmjs\n")
                print(xmjs, file=ofi)
                self.ofi.write("gmjs\n")
                print(gmjs, file=ofi)
                self.ofi.write("pmaxs\n")
                print(pmaxs, file=ofi)
                self.ofi.write("Hops:\n")
                print(self.hop, file=ofi)
                print("cumulative_sum_of_gI", sum_gs, file=ofi)

        if not found_proton:
            print("Error, no protons for indicator")
            raise RuntimeError("calc_indicator 4")

        gI = self.calc_gI(sum_gs)

        self.calc_ind(my_dcom, gI)

        if self.print_all:
            self.ofi.write("gI\n")
            print(gI, file=ofi)
            print("Final location", file=ofi)
            print(self.x_i)

        if self.output_freq:
            if self.step % self.output_freq == 0:
                self._write_log(largest_p, dr, my_dcom[0][np.newaxis, :])
        self.step += 1
        return 0

    def calc_d_weights(self, type_d, x_hms):
        num_d = self.x_d.shape[0]
        w = np.zeros(num_d)
        sum_r = 0
        d_com = np.zeros(3)
        for k in range(num_d):
            rdh0 = self.rxh[type_d[k]]
            for m in range(x_hms[k].shape[0]):
                rdh = np.linalg.norm(self.x_d[k] - x_hms[k][m, :3])
                rdh -= rdh0
                rdh *= rdh
                w[k] += rdh
                sum_r += rdh
        if sum_r > 1e-5:
            w /= sum_r
        else:
            w[:] = 0
            has_h = 0
            for k in range(num_d):
                if x_hms[k].shape[0] > 0:
                    has_h += 1
                    w[k] = 1
            if has_h > 0:
                w[:] /= has_h
            else:
                w[:] = 1 / num_d
        for k in range(num_d):
            d_com += w[k] * self.x_d[k]
        return [d_com]


class Indicator11(Indicator):
    """
    This is equivalent to 4 and was only added because I didn't realize
    that it was the same at the time...

    Do not use it.
    """

    # TODO: Finish this documentation
    def __init__(self):
        Indicator.__init__(self)
        self.donor_com = []
        self.acceptor_com = []

    def calc_indicator(self,
                       x_d,
                       x_as,
                       x_hms,
                       type_d,
                       type_as,
                       d_com,
                       as_com,
                       ofi=None):
        cstr = "{0:0.5f}   {1:0.5f}   {2:0.5f}\n"
        icstr = "{0:9d}   {1:9.5f}   {2:9.5f}   {3:9.5f}\n"

        num_d = len(x_d)
        # Set donor coordinates
        try:
            self.x_d = x_d.copy()
            self.d_com = d_com.copy()[0]
        except RuntimeError("calc_indicator 11"):
            print("Error setting donor coordinates")
            sys.exit()

        # Set acceptor coordinates
        try:
            self.num_acceptors = x_as.shape[0]
            if self.num_acceptors == 0:
                print("Warning: NO ACCEPTOR")
                print("Setting indicator location to donor com")
                self.x_i[:] = self.x_d[:]
                return 1
            self.x_as = np.zeros((self.num_acceptors, 3), dtype=float)
            self.x_as[:, :] = x_as[:, :]
        except RuntimeError("calc_indicator 2"):
            print("Error setting acceptor coordinates")
            sys.exit()

        if self.print_all:
            self.ofi.write("*****STEP %d\n" % self.step)
            self.ofi.write("Donor Coords:\n")
            for k in range(len(x_d)):
                self.ofi.write(cstr.format(*self.x_d[k]))
            self.ofi.write("Acceptor Coords:\n")
            for j in range(self.num_acceptors):
                self.ofi.write(icstr.format(j, *self.x_as[j]))
            self.ofi.write("Proton Coords\n")
            self.ofi.write("Donor COM\n")
            print(d_com, file=ofi)
            print("Acceptor COMs", file=ofi)
            print(as_com)

        gI = 0
        self.x_i = np.zeros(3)
        largest_p = 0
        dr = 0
        self.hop = []
        sum_gs = 0.0
        found_proton = False

        for k in range(num_d):

            # Set hydrogen coordinates
            self.num_h = x_hms[k].shape[0]
            if self.num_h <= 0:
                continue
            found_proton = True
            try:
                self.x_hs = np.zeros((self.num_h, 3), dtype=float)
                self.x_hs = x_hms[k][:, :]
            except RuntimeError("Calc indicator 3"):
                print("Error setting Hydrogen coordinates")
                sys.exit()

            # Initialize the rho parameters
            try:
                rdh0 = self.rxh[type_d[k]]
            except RuntimeError:
                print(
                "Error hashing donor. Is donor in rdh0 list? is only one donor " \
                "passed to subroutine?"
                )
                raise
            pmaxs = np.asarray([rdh0 / (rdh0 + self.rxh[a]) for a in type_as])
            pmj0 = rdh0 / self.rlist

            # Initialize the other arrays
            dims = (self.num_h, self.num_acceptors)
            pmjs = np.zeros(dims, dtype=float)
            xmjs = np.zeros(dims, dtype=float)
            gmjs = np.zeros(dims, dtype=float)

            #Begin the calculations
            for m in range(self.num_h):
                for j in range(self.num_acceptors):
                    pmjs[m, j] = self.calc_pmj(self.x_d[k], self.x_as[j],
                                               self.x_hs[m])
                    if pmjs[m, j] > pmaxs[j]:
                        self.hop.append((m, j, pmjs[m, j], k))
                    if pmjs[m, j] > largest_p:
                        largest_p = pmjs[m, j]
                        dr = np.linalg.norm(self.x_d[k] - self.x_hs[m]) - \
                             np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                    xmjs[m, j] = self.calc_xmj(pmjs[m, j], pmj0, pmaxs[j])
                    gmjs[m, j] = self.calc_gmj(xmjs[m, j])
                    # self.x_i[:] += gmjs[m, j] * (2*x_as[j] - as_com[j] + d_com[0] - x_d[k])
                    # self.x_i[:] += gmjs[m, j] * (x_as[j])
                    self.x_i[:] += gmjs[m, j] * (x_as[j] + self.x_d[k] -
                                                 d_com[0].reshape(3))
            sum_gs += np.sum(gmjs[:])

            if self.print_all:
                print("For donor %d" % k)
                for m in range(self.num_h):
                    self.ofi.write(icstr.format(m, *self.x_hs[m]))
                self.ofi.write("pmjs\n")
                print(pmjs, file=ofi)
                self.ofi.write("xmjs\n")
                print(xmjs, file=ofi)
                self.ofi.write("gmjs\n")
                print(gmjs, file=ofi)
                self.ofi.write("pmaxs\n")
                print(pmaxs, file=ofi)
                self.ofi.write("Hops:\n")
                print(self.hop, file=ofi)
                print("cumulative_sum_of_gI", sum_gs, file=ofi)

        if not found_proton:
            print("Error, no protons for indicator")
            raise RuntimeError("calc_indicator 4")

        gI = self.calc_gI(sum_gs)

        self.calc_ind(d_com, gI)

        if self.print_all:
            self.ofi.write("gI\n")
            print(gI, file=ofi)
            print("Final location", file=ofi)
            print(self.x_i)

        if self.output_freq:
            if self.step % self.output_freq == 0:
                self._write_log(largest_p, dr, d_com[0][np.newaxis, :])
        self.step += 1
        return 0

    def calc_ind(self, d_com, gI):
        self.x_i += d_com[0][:]
        self.x_i /= gI


class VariablePower(Indicator4):
    """
    This indicator exponentiates the g(x) variable using x_min, s, and t.
    It is chosen with Indicator 12 in the code.
    """
    def __init__(self, s=4., t=6.):
        Indicator4.__init__(self)
        self.s = s
        self.t = t
    def set_output_freq(self, freq, prefix=''):
        """
        Initialize variables for writing the xyz and the log file

        Parameters
        ----------
        freq: int
            output frequency
        prefix: str
            file prefix
        """
        try:
            int(freq)
        except TypeError:
            sys.exit("Error: indicator output frequency must be an integer")
        if freq > 0:
            self.output_freq = freq
            if prefix != '':
                self.log_path = prefix + '-' + self.log_path
                self.xyz_path = prefix + '-' + self.xyz_path
            self._lfi = open(self.log_path, 'w')
            self._lfi.write("# Step           rho             dr         minx         maxg\n")
            self._xyz = open(self.xyz_path, 'w')

    def calc_indicator(self, x_d, x_as, x_hms, type_d, type_as, ofi=None):
        """
        This is the main subroutine for calculating the indicator

        Parameters
        ----------
        x_d: ndarray of float
            coordinates with shape [3]
        x_as: ndarray of float
            acceptor coordinates each with shape [j,3]
        x_hms: ndarray of float
            hydrogen coordinates each with shape [m,3]
        type_d: str
            donor type to link with rho parameters
        type_as: list of str
            acceptor types to link with rho parameters
        ofi: file object
            where to print stuff
        """
        cstr = "{0:0.5f}   {1:0.5f}   {2:0.5f}\n"
        icstr = "{0:9d}   {1:9.5f}   {2:9.5f}   {3:9.5f}\n"

        # Set donor coordinates
        try:
            self.x_d[:] = x_d[:]
        except RuntimeError("calc_indicator 1"):
            print("Error setting donor coordinates")
            sys.exit()

        # Set acceptor coordinates
        try:
            self.num_acceptors = x_as.shape[0]
            if self.num_acceptors == 0:
                print("Warning: NO ACCEPTOR")
                print("Setting indicator location to donor coordinates")
                self.x_i[:] = self.x_d[:]
                return 1
            self.x_as = np.zeros((self.num_acceptors, 3), dtype=float)
            self.x_as[:, :] = x_as[:, :]
        except RuntimeError("calc_indicator 2"):
            print("Error setting acceptor coordinates")
            sys.exit()

        # Set hydrogen coordinates
        self.num_h = x_hms.shape[0]
        if self.num_h <= 0:
            print("Error, no protons for indicator")
            raise RuntimeError("calc_indicator 4")
        try:
            self.x_hs = np.zeros((self.num_h, 3), dtype=float)
            self.x_hs = x_hms[:, :]
        except RuntimeError("Calc indicator 3"):
            print("Error setting Hydrogen coordinates")
            sys.exit()

        # Initialize the rho parameters
        try:
            rdh0 = self.rxh[type_d]
        except RuntimeError:
            print("Error hashing donor. Is donor in rdh0 list? is only one"
                  " donor passed to subroutine?")
            raise
        pmaxs = np.asarray([rdh0 / (rdh0 + self.rxh[a]) for a in type_as])
        pmj0 = rdh0 / self.rlist

        # Initialize the other arrays
        dims = (self.num_h, self.num_acceptors)
        pmjs = np.zeros(dims, dtype=float)
        xmjs = np.zeros(dims, dtype=float)
        gmjs = np.zeros(dims, dtype=float)
        xmjes= np.zeros(dims, dtype=float)
        self.hop = []

        #Begin the calculations
        largest_p = 0
        dr = 0
        for m in range(self.num_h):
            for j in range(self.num_acceptors):
                pmjs[m, j] = self.calc_pmj(self.x_d, self.x_as[j], self.x_hs[m])
                if pmjs[m, j] > pmaxs[j]:
                    self.hop.append((m, j, pmjs[m, j]))
                if pmjs[m, j] > largest_p:
                    largest_p = pmjs[m, j]
                    dr = np.linalg.norm(self.x_d - self.x_hs[m]) - \
                         np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                xmjs[m, j] = self.calc_xmj(pmjs[m, j], pmj0, pmaxs[j])
                #gmjs[m, j] = self.calc_gmj(xmjs[m, j])
        xmjes = self.calc_xmje(xmjs, xmjes, self.s, self.t)
        for m in range(self.num_h):
            for j in range(self.num_acceptors):
                gmjs[m, j] = self.calc_gmj(xmjs[m, j], xmjes[m, j])
        minx, maxg = self.calc_vars(xmjs, gmjs)
        gI = self.calc_gI(gmjs)
        self.x_i[:] = self.x_d[:]
        for j in range(self.num_acceptors):
            for m in range(self.num_h):
                self.x_i[:] += gmjs[m, j] * x_as[j]
        self.x_i *= 1. / gI

        if self.print_all:
            self.ofi.write("Detailed Stats\n")
            self.ofi.write("Donor Coords:\n")
            self.ofi.write(cstr.format(*self.x_d))
            self.ofi.write("Acceptor Coords:\n")
            for j in range(self.num_acceptors):
                self.ofi.write(icstr.format(j, *self.x_as[j]))
            self.ofi.write("Proton Coords\n")
            for m in range(self.num_h):
                self.ofi.write(icstr.format(m, *self.x_hs[m]))
            self.ofi.write("pmjs\n")
            print(pmjs, file=ofi)
            self.ofi.write("xmjs\n")
            print(xmjs, file=ofi)
            self.ofi.write("gmjs\n")
            print(gmjs, file=ofi)
            self.ofi.write("gI\n")
            print(gI, file=ofi)
            self.ofi.write("Hops:\n")
            print(self.hop, file=ofi)
        if self.output_freq:
            if self.step % self.output_freq == 0:
                self._write_log(largest_p, dr, minx, maxg, coords=self.x_d.reshape(-1, 3))
        self.step += 1
        return 0

    @staticmethod
    @jit(nopython=True)
    def calc_vars(amx, gmjs):
        """provides minimum variables for analysis"""
        minx = 1
        maxg = 0
        for m in range(len(amx)):
            for j in range(len(amx[0])):
                if amx[m,j] >= 1:
                    nothing = 0
                elif amx[m, j] < 0:
                    minx = 0
                    maxg = gmjs[m, j]
                elif amx[m, j] < minx and amx[m, j] >=0:
                    minx = amx[m, j]
                    maxg = gmjs[m, j]
        return minx, maxg

    def _write_log(self, p, dr, minx, maxg, coords=None):
        """
        Write the results of a step
        Write the rho and dr coordinates
        If a matrix of coordinates is present, write them to the xyz file

        Parameters
        ----------
        :param p:
        :param dr:
        :param minx: float
        :param maxg: float
        :param coords:
        :return:
        """
        self._lfi.write('{0:10d}  {1:10.6f}   {2:10.6f}   {3:10.6f}   {4:10.6f}\n'.format(
            self.step, p, dr, minx, maxg))

        if coords is None:
            return
        #if not coords.any():
        #    return

        natoms = coords.shape[0]

        try:
            self._xyz.write('%d\n' % self.max_xyz_atoms)
        except:
            sys.exit("Error writing number of atoms")
        self._xyz.write('\n')

        xyz_str = "{3}     {0:12.6}     {1:12.6f}     {2:12.6f}\n"
        els = ["Ti", "V ", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge"]
        try:
            for i in range(natoms):
                self._xyz.write(xyz_str.format(*coords[i], els[i]))
            for j in range(i + 1, self.max_xyz_atoms):
                self._xyz.write(xyz_str.format(0., 0., 0., els[j]))
        except IOError:
            print("Error writing coords")
            raise
        self._lfi.flush()
        return
    @staticmethod
    @jit(nopython=True)
    def calc_xmje(amx, xmjes, s, t):
        """calculate the variable B for g(x). This is the exponent that deals
        with how much power must be added to smoothing function

        Parameters
        ----------
        amx: xmjs as ndrray
        xmjes: empty ndarray for exponents
        s: float for s parameter in exponent
        t: float for t parameter in exponent

        Returns
        -------
        xmjes: ndarray
        """
        min =1
        for m in range(len(amx)):
            for j in range(len(amx[0])):
                if amx[m, j] >= 1:
                    xmjes[m, j] = 1
                elif amx[m, j] < 0:
                    xmjes[m, j] = 1
                    min = 0
                elif amx[m, j] < min and amx[m, j] >=0:
                    min = amx[m,j]
        for m in range(len(amx)):
            for j in range(len(amx[0])):
                if amx[m, j] >= 0 and amx[m, j] < 1:
                    xmjes[m, j] = float(t + (amx[m, j] - min)*s)
                    """The power level goes from 1 to 10 for B"""
        return xmjes

    @staticmethod
    @jit(nopython=True)
    def calc_gmj(xmj, xmjes):

        if 1 <= xmj:
            gmj = 0
        elif xmj < 0:
            gmj = 1
        else:
            gmj = (-6 * xmj**5 + 15 * xmj**4 - 10 * xmj**3 + 1)**xmjes
        return gmj

class Softmax(Indicator4):
    def __init__(self, a=0.05, zmax=30):
        Indicator4.__init__(self)
        self.a = a
        self.zmax = zmax
    pass

    def calc_indicator(self, x_d, x_as, x_hms, type_d, type_as, ofi=None):
        """
        This is the main subroutine for calculating the indicator

        Parameters
        ----------
        x_d: ndarray of float
            coordinates with shape [3]
        x_as: ndarray of float
            acceptor coordinates each with shape [j,3]
        x_hms: ndarray of float
            hydrogen coordinates each with shape [m,3]
        type_d: str
            donor type to link with rho parameters
        type_as: list of str
            acceptor types to link with rho parameters
        ofi: file object
            where to print stuff
        """
        cstr = "{0:0.5f}   {1:0.5f}   {2:0.5f}\n"
        icstr = "{0:9d}   {1:9.5f}   {2:9.5f}   {3:9.5f}\n"

        # Set donor coordinates
        try:
            self.x_d[:] = x_d[:]
        except RuntimeError("calc_indicator 1"):
            print("Error setting donor coordinates")
            sys.exit()

        # Set acceptor coordinates
        try:
            self.num_acceptors = x_as.shape[0]
            if self.num_acceptors == 0:
                print("Warning: NO ACCEPTOR")
                print("Setting indicator location to donor coordinates")
                self.x_i[:] = self.x_d[:]
                return 1
            self.x_as = np.zeros((self.num_acceptors, 3), dtype=float)
            self.x_as[:, :] = x_as[:, :]
        except RuntimeError("calc_indicator 2"):
            print("Error setting acceptor coordinates")
            sys.exit()

        # Set hydrogen coordinates
        self.num_h = x_hms.shape[0]
        if self.num_h <= 0:
            print("Error, no protons for indicator")
            raise RuntimeError("calc_indicator 4")
        try:
            self.x_hs = np.zeros((self.num_h, 3), dtype=float)
            self.x_hs = x_hms[:, :]
        except RuntimeError("Calc indicator 3"):
            print("Error setting Hydrogen coordinates")
            sys.exit()

        # Initialize the rho parameters
        try:
            rdh0 = self.rxh[type_d]
        except RuntimeError:
            print("Error hashing donor. Is donor in rdh0 list? is only one"
                  " donor passed to subroutine?")
            raise
        pmaxs = np.asarray([rdh0 / (rdh0 + self.rxh[a]) for a in type_as])
        pmj0 = rdh0 / self.rlist

        # Initialize the other arrays
        dims = (self.num_h, self.num_acceptors)
        pmjs = np.zeros(dims, dtype=float)
        xmjs = np.zeros(dims, dtype=float)
        gmjs = np.zeros(dims, dtype=float)
        zmjs = np.zeros(dims, dtype=float)
        exmjs = np.zeros(dims, dtype=float)
        sms = np.zeros(dims, dtype=float)
        self.hop = []

        #Begin the calculations
        largest_p = 0
        dr = 0
        for m in range(self.num_h):
            for j in range(self.num_acceptors):
                pmjs[m, j] = self.calc_pmj(self.x_d, self.x_as[j], self.x_hs[m])
                if pmjs[m, j] > pmaxs[j]:
                    self.hop.append((m, j, pmjs[m, j]))
                if pmjs[m, j] > largest_p:
                    largest_p = pmjs[m, j]
                    dr = np.linalg.norm(self.x_d - self.x_hs[m]) - \
                         np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                xmjs[m, j] = self.calc_xmj(pmjs[m, j], pmj0, pmaxs[j])
                gmjs[m, j] = self.calc_gmj(xmjs[m, j])
        zmjs = self.calc_zmj(gmjs, zmjs)
        sms = self.calc_sm(zmjs, xmjs, exmjs, sms, gmjs)
        sI = self.calc_sI(sms)
        gI = self.calc_gI(gmjs)
        self.x_i[:] = self.x_d[:]
        for j in range(self.num_acceptors):
            for m in range(self.num_h):
                self.x_i[:] += sms[m, j] * x_as[j]
        self.x_i *= 1. / sI

        if self.print_all:
            self.ofi.write("Detailed Stats\n")
            self.ofi.write("Donor Coords:\n")
            self.ofi.write(cstr.format(*self.x_d))
            self.ofi.write("Acceptor Coords:\n")
            for j in range(self.num_acceptors):
                self.ofi.write(icstr.format(j, *self.x_as[j]))
            self.ofi.write("Proton Coords\n")
            for m in range(self.num_h):
                self.ofi.write(icstr.format(m, *self.x_hs[m]))
            self.ofi.write("pmjs\n")
            print(pmjs, file=ofi)
            self.ofi.write("xmjs\n")
            print(xmjs, file=ofi)
            self.ofi.write("gmjs\n")
            print(gmjs, file=ofi)
            self.ofi.write("sms\n")
            print(sms, file=ofi)
            self.ofi.write("sI\n")
            print(sI, file=ofi)
            self.ofi.write("Hops:\n")
            print(self.hop, file=ofi)
        if self.output_freq:
            if self.step % self.output_freq == 0:
                self._write_log(largest_p, dr, coords=self.x_d.reshape(-1, 3))
        self.step += 1
        return 0

    @staticmethod
    @jit
    def calc_zmj(gmjs, zmjs):
        """
        Calculate the softmax function zmj parameter

        Parameters
        ----------
        gmjs: array
            smoothing function parameters

        Returns
        ----------
        zmjs: array
        """
        for m in range(len(gmjs)):
            for j in range(len(gmjs[0])):
                tempz=gmjs[m][j]/(1.0000000000001-gmjs[m][j])
                if tempz > 30:
                    zmjs[m][j] = 30
                else:
                    zmjs[m][j] = tempz
        return zmjs
    @staticmethod
    @jit
    def calc_sm(zmjs, xmjs, exmjs, sms, gmjs):
        """
        Calculates the Softmax function sz

        Paremeters
        ----------
        zmjs: the calculated numerator vectors
        xmjs: the calcualted xrho values
        sms: empty array in dims of zmjs

        Returns
        ----------
        sms: array full of softmax parameters
        """
        for m in range(len(zmjs)):
            for j in range(len(zmjs[0])):
                exmjs[m][j] = np.exp(0.05*zmjs[m][j])-1
        denom = np.sum(exmjs)
        for m in range(len(zmjs)):
            for j in range(len(zmjs[0])):
                sms[m][j] = ((np.exp(0.05*zmjs[m][j])-1)/(denom + 0.0000000000000001))*gmjs[m][j]
        return sms
    @staticmethod
    @jit
    def calc_sI(sms):
        """
        Calculate the normalization constant sI

        Parameters
        ----------
        :param sms: the projection vectors
        :type sms: np.ndarray
        :return: the normalization constant
        :rtype: np.float
        """
        return 1 + np.sum(sms)

class MCEC(Indicator4):
    """
    Implementation of the mCEC.
    This implementation of mCEC has indicator 4 as base class for switching
    topology. Then the mCEC stuff sits right on top of it.
    """

    def __init__(self, switching='chakrabarti'):
        Indicator4.__init__(self)
        if switching == 'chakrabarti':
            self.switch = chakrabarti_switching
        elif switching == 'fos':
            self.switch = self.fos
            print("Error, not implemented")
            sys.exit()
        else:
            print("Improper value for switching function. Should be "
                  "'chakrabarti' or 'fos'")
            sys.exit()

        self.switch = np.vectorize(self.switch)
        self.m_acc_weight = {'OT': 2, 'SOT': 2, 'CLA': 0, 'OC': 0}
        self.rsw = 1.40
        self.dsw = 0.04
        self.x_mcec = np.asarray([0.00, 0.00, 0.00])
        self.correction_groups = []
        self.correction_weights = []

    def calc_mcec(self, rH, rXj, acc_types, correction_groups=None):
        """
        Main loop for calculating the mCEC location.

        The result is stored in self.x_mcec.

        Parameters
        ----------
        rH: ndarray with shape(m,3)
            The positions of the hydrogens
        rXj: ndarray of float with shape (j,3)
            The locations of the acceptors
        acc_types: list of str with len (j)
            The atom type corresponding to an the Jth acceptor
        correction_groups: list of lr

        Returns
        -------

        """
        if len(acc_types) != rXj.shape[0]:
            print("Error, number of acceptor types does not equal"
                  "the number of acceptor coordinates")
        if rH.size == 0:
            print("Error, no hydrogen coordinates found")
            raise IndexError
        if rXj.size == 0:
            print("Error, no acceptor coordinates found")
            raise IndexError
        self.get_weight_vector(acc_types)
        self.x_mcec[:] = calc_mcec_location(rH, rXj, self.acc_weights, self.rsw,
                                            self.dsw)
        # self.x_mcec[:] = self.calc_mcec_location(rH, rXj, self.acc_weights,
        #                                          self.switch, self.rsw, self.dsw)
        print("MCEC before correction", self.x_mcec)
        if correction_groups:
            self.x_mcec[:] += self.calc_mcec_correction(rH, correction_groups)
        print("Final mCEC", self.x_mcec)

    def calc_mcec_correction(self, rH, rGroups, verbose=True):
        """
        Calculate the correction term for the mCEC.

        Currently the max function is used instead of the nondifferentiable
        max function due to numerical issues.

        Parameters
        ----------
        rH: ndarray of float with shape(m,3)
            The positions of the hydrogens
        rGroups: list of ndarrays with shape(m,3)
            The position of the groups.
            Example:
            ([[1.1 1.1 1.1],
              [1.2 1.2 1.2]],
             [[2.1 2.1 2.1],
              [2.2 2.2 2.2],
              [2.3 2.3 2.3]])

        Returns
        -------
        ndarray of float with shape(3)

        """
        num_groups = len(self.correction_groups)
        correction = np.asarray([0., 0., 0.])
        my_correction = np.asarray([0., 0., 0.])
        if num_groups != len(rGroups):
            print(
                "Error, the number of groups found does not equal the number of groups parsed"
            )
            raise LookupError
        for g in range(num_groups):
            my_correction[:] = 0.
            group_length = len(self.correction_groups[g])
            group_diff_max = np.empty(group_length)
            # Calculate the array of switching functions
            for x in range(group_length):
                dists = rH - rGroups[g][x]
                dists = np.linalg.norm(dists, axis=1)
                dists = chakrabarti_switching(dists, self.rsw, self.dsw)
                # The differentiable maximum function is disabled.
                # group_diff_max[x] = self.diff_max(dists)
                group_diff_max[x] = dists.max()
            print(group_diff_max)
            for l, k in permutations(range(group_length), 2):
                my_correction += group_diff_max[k] * (rGroups[g][l] -
                                                      rGroups[g][k])
            my_correction *= self.correction_weights[g]
            correction += my_correction
        if verbose:
            print('Correciton amount', correction)
        return correction

    def get_weight_vector(self, types):
        """
        Return an array of weights for each exceptor in list 'types'

        Parameters
        ----------
        types: list of str
            list of atom types to lookup weights for

        Returns
        -------
        ndarray of floats

        Exceptions
        ----------
        LookupError: Could not find the type in the acceptor type dictionary
        """
        num_acc = len(types)
        self.acc_weights = np.zeros(num_acc, dtype=float)

        for i in range(num_acc):
            try:
                self.acc_weights[i] = self.m_acc_weight[types[i]]
            except LookupError:
                print("Error looking up acceptor %d, type not found" % i)
                raise

    @staticmethod
    @jit(nopython=True)
    def diff_max(results, power=15):
        """
        Differentiable maximum function. Given a list of floats,
        calculate the differentiable maximum function

        Parameters
        ----------
        results: ndarray of floats
            array of floats to exponentiate and sum

        Returns
        -------
        float
        """
        a = results**power
        b = a * results
        if (a[:] == np.nan).any():
            a[a == np.nan] == 0.0
        if (b[:] == np.nan).any():
            b[b == np.nan] == 0.0
        return b.sum() / a.sum()

    # @staticmethod
    # def calc_mcec_location(rH, rXj, w, switch, rsw, dsw):
    #     """
    #     Returns the mcec location from equation 6 of
    #     J. Phys. Chem. A, Vol. 110, 2006
    #
    #     This is the mCEC location without the correction term
    #
    #     :param rH: np.ndarray of floats with size (n,3) where n is
    #     number of hydrogens. hydrogen locations
    #     :param rXj: np.ndarray of floats with size (J,3) where J is
    #     number of acceptors. acceptor locations
    #     :param w: np.ndarray of integers with size J representing
    #     the minimum protonatied state of the acceptor
    #     :param switch: vectorized switching function that takes a scalar
    #      distance. Must accept numpy arrays.
    #     :return: zeta: The mcec without the correction term
    #     """
    #
    #     # hydrogen and weighted acceptors
    #     zeta = np.sum(rH, axis=0)
    #     zeta -= np.dot(w, rXj)
    #
    #     num_m = rH.shape[0]
    #     num_j = rXj.shape[0]
    #     slow = False # Is 2 seconds slower for 25 second job...
    #     #_Slow way to calculate zeta
    #     if slow:
    #         for m in range(num_m):
    #             for j in range(num_j):
    #                 displacement = rH[m] - rXj[j]
    #                 distance = np.linalg.norm(displacement)
    #                 factor = switch(distance, rsw, dsw)
    #                 zeta -= factor * displacement
    #     else:
    #         # pairwise distances and switching functions
    #         rHXj = np.zeros((rXj.shape[0], rH.shape[0], 3))
    #         rHXj[:] = rH[:]
    #         rHXj = np.transpose(rHXj, (1, 0, 2)) - rXj
    #         zeta -= np.tensordot(rHXj, switch(np.linalg.norm(rHXj, axis=2), rsw, dsw), [(0,1),(0,1)])
    #     return zeta

    # @staticmethod
    # def chakrabarti_switching(d, rsw, dsw):
    #     """
    #     Chakrabarti switching function from
    #     Konig et all
    #     J. Phys. Chem. A. Vol 110, No.2
    #     :param d: real distance
    #     :param rsw: real midpoint of switching function
    #     :param dsw: real slope of switching function
    #     :return:
    #     """
    #     return (1+np.exp((d-rsw)/dsw))**-1

    @staticmethod
    @jit(nopython=True)
    def fos(x):
        """
        Our fifth order spline

        Parameters
        ----------
        :param x:
        :return:
        """
        return -6 * x**5 + 15 * x**4 - 10 * x**3 + 1

class EPI(Indicator4):
    def __init__(self, c=0.0, ro=1.3, a=0.129):
        Indicator4.__init__(self)
        self.a = a # NOte in the equation, they say that a is a parameter but
                   # in the equation it says d. Here we set d to a
        self.d = self.a
        self.ro = ro
        self.c = c
        self.correction_groups = None
        # Yes this is the EPi but we can just use this as the
        # holder for the variable to reuse as much code as possible
        self.x_mcec = np.zeros(3)
    def calc_mcec(self, rH, rXj, acc_types, correction_groups=None):
        """
        Main loop for calculating the EPI location.

        The result is stored in self.x_mcec.

        Parameters
        ----------
        rH: ndarray with shape(m,3)
            The positions of the hydrogens
        rXj: ndarray of float with shape (j,3)
            The locations of the acceptors
        acc_types: list of str with len (j)
            The atom type corresponding to an the Jth acceptor
        correction_groups: list of lr

        Returns
        -------

        """
        if rH.size == 0:
            print("Error, no hydrogen coordinates found")
            raise IndexError
        if rXj.size == 0:
            print("Error, no acceptor coordinates found")
            raise IndexError
        self.x_mcec[:] = calc_epi_location(rH, rXj, self.c, self.ro, self.d)
        print("Final EPI", self.x_mcec)

@jit
def calc_epi_location(rH, rXj, c, r0, d):
    """
    Here we use i for oxygens and a for hydrogen indices like in the
    paper
    Parameters
    ----------
    rH
    rXj
    c
    r0
    ao

    Returns
    -------

    """
    num_hyd = rH.shape[0]
    num_acc = rXj.shape[0]
    epi = np.zeros(3)
    sum_W = 0
    for i in range(num_acc):
        # Distance between ith oxygen and all of the hydrogens
        ria = np.linalg.norm(rXj[i] - rH, axis=1)
        n_i = calc_virtual_site(rH, ria, num_hyd, r0, d)
        z_i = c*n_i + (1-c)*rXj[i]
        W_i = calc_Wi(ria, num_hyd)
        sum_W += W_i
        epi += W_i * z_i
    epi /= sum_W
    return epi

@jit(nopython=True)
def calc_Wi(ria, num_hyd):
    psi_i = 0
    for a in range(num_hyd):
        psi_i += epi_spline_S(ria[a])
    return epi_spline_W(psi_i)

@jit(nopython=True)
def calc_virtual_site(rH, ria, num_hyd, r0, d):
    """
    Calculate the virtual site (eta_i) from equation 15 by summation
    of gaussians phi_ia multiplied by the coordinates of its hydrogen a
    Parameters
    ----------
    rH: ndarray
        mx3 array of hydrogen coordinates
    ria: ndarray
        m array of distances from an oxygen to the a'th hydrogen
    num_hyd: int
        this is m, the number of hydrogens
    r0: float
        the r0 parameter for the gaussian function
    d: float
        the d parameter for the gaussian function

    Returns
    -------
    virt_site: ndarray
        vector of floats, size 3

    """
    virt_site = np.zeros(3)
    sum_phi = 0.0
    for a in range(num_hyd):
        phi_ia = epi_gauss(ria[a], r0, d)
        virt_site += phi_ia * rH[a]
        sum_phi += phi_ia
    virt_site /= sum_phi
    return virt_site

@jit(nopython=True)
def epi_gauss(x, r0, d):
    """
    The EPI Gaussian function (equation 16)
    Parameters
    ----------
    x: float
    ro: float
    d: float

    Returns
    -------
    float
    """
    return np.exp(-(x - r0)**2 / d**2)

@jit(nopython=True)
def epi_spline_S(x):

    a = 12.0
    b = 13.2
    d = b - a
    p = (2*a + b) / 3
    q = (a + 2*b) / 3

    if x < a:
        return 1
    elif x >= a and x < p:
        return -9*(x - a)**3 / (2 * d**3) + 1
    elif x >= p and x < q:
        return 9 * (x - p)**3 / (d**3) -\
               9 * (x - p)**2 / (2 * d**2) -\
               3*(x-p)/(2*d) + 5/6.
    elif x >= q and x < b:
        return -9*(x - b)**3 / (2 * d**3)
    return 0

@jit(nopython=True)
def epi_spline_W(x):
    """
    Calculate the EPI spline function (equation 31) for the
    equation 19 in the EPI paper.
    This outputs a float between 1 and 0 for numbers within the range
    of a=2 and b=3, respectively.

    The a, b, c, and d parameters are hardcoded.

    CHECK IF  9(d-p)**2 should be 9(x-p)**2
    Parameters
    ----------
    x: float

    Returns
    -------
    float
    """
    a = 2
    b = 3
    d = b - a
    p = (2 * a + b) / 3
    q = (a + 2 * b) / 3

    if x < a:
        return 0.
    elif x >= a and x < p:
        return 9 * (x - a) ** 3 / (2 * d ** 3)
    elif x >= p and x < q:
        return -9 * (x - p) ** 3 / d ** 3 + \
               9 * (x - p) ** 2 / (2 * d ** 2) + \
               3 * (x - p) / (2 * d) + 1 / 6.
    elif x >= q and x < b:
        return 9 * (x - b) ** 3 / 2 * d ** 3 + 1
    return 1.


@jit(nopython=True, parallel=True)
def chakrabarti_switching(d, rsw, dsw):
    """
    Chakrabarti switching function from
    Konig et all
    J. Phys. Chem. A. Vol 110, No.2

    Parameters
    ----------
    d: real
       distance
    rsw: real
        midpoint of switching function
    dsw: real
        slope of switching function

    Returns
    -------
    float
    """
    return (1 + np.exp((d - rsw) / dsw))**-1


@jit
def calc_mcec_location(rH, rXj, w, rsw, dsw, verbose=False):
    """
    Returns the mcec location from equation 6 of
    J. Phys. Chem. A, Vol. 110, 2006

    This is the mCEC location without the correction term

    Parameters
    ----------
    rH: np.ndarray of floats with size (n,3) where n is
    number of hydrogens.
         hydrogen locations
    rXj: np.ndarray of floats w ith size (J,3) where J is
    number of acceptors.
        acceptor locations
    w: np.ndarray of int with size J
        The reference protonation state of each acceptor

    Returns
    -------
    zeta: The mcec without the correction term
    """

    # hydrogen and weighted acceptors
    zeta = np.sum(rH, axis=0)
    if verbose:
        print('sum_hydrogen', zeta)
    zeta -= np.dot(w, rXj)
    print('after subtracting acc', zeta)
    num_m = rH.shape[0]
    num_j = rXj.shape[0]
    slow = False  # Is 2 seconds slower for 25 second job...
    #_Slow way to calculate zeta
    if slow:
        for m in range(num_m):
            for j in range(num_j):
                displacement = rH[m] - rXj[j]
                distance = np.linalg.norm(displacement)
                factor = chakrabarti_switching(distance, rsw, dsw)
                zeta -= factor * displacement
    else:
        # pairwise distances and switching functions
        rHXj = np.zeros((rXj.shape[0], rH.shape[0], 3))
        rHXj[:] = rH[:]
        rHXj = np.transpose(rHXj, (1, 0, 2)) - rXj
        zeta -= np.tensordot(
            rHXj, chakrabarti_switching(np.linalg.norm(rHXj, axis=2), rsw, dsw),
            [(0, 1), (0, 1)])
    return zeta
