#!/usr/bin/env python3
"""
This file contains classes for calculating the position of the indicator
with Numpy
"""
__author__ = 'Adam Duster'
__copyright__ = ''
__credits__ = ['Adam Duster']
__license__ = 'CC-BY-SA'
__version__ = '0.1'
__email__ = 'adam.duster@ucdenver.edu'
__status__ = 'Development'
import sys
import numpy as np
from math import exp
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
        self.rxh = {
            'OT': 1.0,
            'SOT': 1.0,
            'CLA': 1.4,
            'OC': 1.0
        }
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
        self.xyz_path = 'indicator.xyz'
        # Where to print debug information?
        self.ofi = sys.stdout
        # Log file object
        self._lfi = None
        self._xyz = None
        self.max_xyz_atoms = 5
        # This step
        self.step = 0

    def set_output_freq(self, freq, prefix=''):
        """
        Initialize variables for writing the xyz and the log file
        :param freq: int - output frequency
        :param prefix: str - file prefix
        :return:
        """
        try:
            int(freq)
        except:
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
        :return:
        """
        self.hop = []

    def _write_log(self, p, dr, coords=None):
        """
        Write the results of a step
        Write the rho and dr coordinates
        If a matrix of coordinates is present, write them to the xyz file
        :param p:
        :param dr:
        :param coords:
        :return:
        """
        self._lfi.write('{0:10d}  {1:10.6f}   {2:10.6f}\n'.format(self.step, p, dr))

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
                self._xyz.write(xyz_str.format(0., 0., 0., els[j] ))
        except:
            sys.exit("Error writing coords")


    def add_rdh0(self, rdh, atom_type : str):
        """
        Add a rho parameter and atom type to the list of parameters
        :param rdh:
        :param atom_type:
        :return:
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
                  "rdh".format( atom_type, rdh))
        return

    def print_rdh0(self, ofi=None):
        """
        Output data for double checking
        :param ofi:
        :return:
        """
        print("Atom_Type  rDH", file=ofi)
        for key, rdh in self.rxh:
            print("{0:8}    {0:5.3f}".format(key, rdh), file=ofi)
        return

    def calc_indicator(self, x_d, x_as, x_hms, type_d, type_as, ofi=None):
        """
        This is the main subroutine for calculating the indicator
        :param x_d: ndarray float coordinates with shape [3]
        :param x_as: ndarray of acceptor coordinates each with shape [j,3]
        :param x_hms: ndarray of hydrogen coordiantes each with shape [m,3]
        :param type_d: string of donor types to link with rho parameters
        :param type_as: list of strings of acceptor types to link with rho parameters
        :param ofi: where to print stuff
        :return:
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
                print("WARNING: NO ACCEPTOR")
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
            "Error hashing donor. Is donor in rdh0 list? is only one donor " \
            "passed to subroutine?"
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
                pmjs[m,j] = self.calc_pmj(self.x_d, self.x_as[j], self.x_hs[m])
                if pmjs[m, j] > pmaxs[j]:
                    self.hop.append((m, j, pmjs[m, j]))
                if pmjs[m, j] > largest_p:
                    largest_p = pmjs[m, j]
                    dr = np.linalg.norm(self.x_d - self.x_hs[m]) - \
                         np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                xmjs[m,j] = self.calc_xmj(pmjs[m,j], pmj0, pmaxs[j])
                gmjs[m,j] = self.calc_gmj(xmjs[m,j])

        gI = self.calc_gI(gmjs)
        self.x_i[:] = self.x_d[:]
        for j in range(self.num_acceptors):
            for m in range(self.num_h):
                self.x_i[:] += gmjs[m,j] * x_as[j]
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
    def calc_pmj(x_d, x_aj, x_hm):
        """
        calculate the variable p_mj [ rho_mj ]
        this is the projection of the D-H vec onto the D-A vec
        :param x_d: coordinates of donor (np.float array [3] )
        :type x_d: np.float[3]
        :param x_aj: coordinates of acceptor j (np.float array [3] )
        :type x_aj: np.float
        :param x_hm: coordinates of hydrogen m (np.float array [3] )
        :type x_hm: np.float
        :rtype x_d: np.float
        :return:
        """
        r_dhm = x_hm - x_d
        r_daj = x_aj - x_d
        return np.dot(r_dhm, r_daj) / np.linalg.norm(r_daj) ** 2

    @staticmethod
    def calc_xmj(pmj, pmj0, pmax, debug=False):
        """
        calculate the variable x(p_mj) [ x(rho_mj) ]
        :param pmj: projection scalar
        :param pmj0: scaling parameter parameter
        :param pmax: equilibrium bond constant ratio
        :type pmj: float
        :type pmj0: float
        :type pmax: float
        :return: x(p_mj)
        :rtype: float
        """
        # if debug:
        #     print(pmj, pmj0, pmax)
        return 1 - (pmj - pmj0) / (pmax - pmj0)

    @staticmethod
    def calc_gmj(xmj):

        if 1 <= xmj:
            gmj = 0.
        elif xmj < 0:
            gmj = 1
        else:
            gmj = -6*xmj**5 + 15*xmj**4 -10*xmj**3 + 1
        return gmj

    @staticmethod
    def calc_gI(gmjs):
        """
        Calculate the normalization constant gI
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
                    pmjs[m,j] = self.calc_pmj(self.x_d[k], self.x_as[j], self.x_hs[m])
                    if pmjs[m, j] > pmaxs[j]:
                        self.hop.append((m, j, pmjs[m, j], k))
                    if pmjs[m, j] > largest_p:
                        largest_p = pmjs[m, j]
                        dr = np.linalg.norm(self.x_d[k] - self.x_hs[m]) - \
                             np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                    xmjs[m,j] = self.calc_xmj(pmjs[m,j], pmj0, pmaxs[j])
                    gmjs[m,j] = self.calc_gmj(xmjs[m,j])
                    # self.x_i[:] += gmjs[m, j] * (2*x_as[j] - as_com[j] + d_com[0] - x_d[k])
                    # self.x_i[:] += gmjs[m, j] * (x_as[j])
                    self.x_i[:] += gmjs[m, j] * (as_com[j])
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
                self._write_log(largest_p, dr, d_com[0][np.newaxis,:])
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
            gmj = -6*xmj**5 + 15*xmj**4 -10*xmj**3 + 1
        return gmj * np.exp(gmj)

    def calc_ind(self, d_com, gI):
        self.x_i += d_com[0][:] * np.e
        self.x_i /= gI

    @staticmethod
    def calc_gI(gmjs):
        """
        Calculate the normalization constant gI
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

    def calc_indicator(self, x_d, x_as, x_hms, type_d, type_as, d_com, as_com,
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
                    pmjs[m,j] = self.calc_pmj(self.x_d[k], self.x_as[j], self.x_hs[m])
                    if pmjs[m, j] > pmaxs[j] and not intra_p:
                        self.hop.append((m, j, pmjs[m, j], k, False))
                    if pmjs[m, j] > largest_p:
                        largest_p = pmjs[m, j]
                        dr = np.linalg.norm(self.x_d[k] - self.x_hs[m]) - \
                             np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                    xmjs[m,j] = self.calc_xmj(pmjs[m,j], pmj0, pmaxs[j])
                    gmjs[m,j] = self.calc_gmj(xmjs[m,j])
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
                    my_p_don = rah / (rah + np.linalg.norm(self.x_hs[m] - self.x_d[k]))
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

    def calc_indicator(self, x_d, x_as, x_hms, type_d, type_as, d_com, as_com,
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
                    pmjs[m,j] = self.calc_pmj(self.x_d[k], self.x_as[j], self.x_hs[m])
                    if pmjs[m, j] > pmaxs[j]:
                        self.hop.append((m, j, pmjs[m, j], k))
                    if pmjs[m, j] > largest_p:
                        largest_p = pmjs[m, j]
                        dr = np.linalg.norm(self.x_d[k] - self.x_hs[m]) - \
                             np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                    xmjs[m,j] = self.calc_xmj(pmjs[m,j], pmj0, pmaxs[j])
                    gmjs[m,j] = self.calc_gmj(xmjs[m,j])
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
                self._write_log(largest_p, dr, my_dcom[0][np.newaxis,:])
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
    This implementation of the indicator is the one I develped but
    the distance between the donor center of mass and the kth donor
    are added to the final result.
    """
    # TODO: Finish this documentation
    def __init__(self):
        Indicator.__init__(self)
        self.donor_com = []
        self.acceptor_com = []


    def calc_indicator(self, x_d, x_as, x_hms, type_d, type_as, d_com, as_com,
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
                    pmjs[m,j] = self.calc_pmj(self.x_d[k], self.x_as[j], self.x_hs[m])
                    if pmjs[m, j] > pmaxs[j]:
                        self.hop.append((m, j, pmjs[m, j], k))
                    if pmjs[m, j] > largest_p:
                        largest_p = pmjs[m, j]
                        dr = np.linalg.norm(self.x_d[k] - self.x_hs[m]) - \
                             np.linalg.norm(self.x_as[j] - self.x_hs[m] )
                    xmjs[m,j] = self.calc_xmj(pmjs[m,j], pmj0, pmaxs[j])
                    gmjs[m,j] = self.calc_gmj(xmjs[m,j])
                    # self.x_i[:] += gmjs[m, j] * (2*x_as[j] - as_com[j] + d_com[0] - x_d[k])
                    # self.x_i[:] += gmjs[m, j] * (x_as[j])
                    self.x_i[:] += gmjs[m, j] * (x_as[j] + self.x_d[k] - d_com[0].reshape(3))
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
                self._write_log(largest_p, dr, d_com[0][np.newaxis,:])
        self.step += 1
        return 0

    def calc_ind(self, d_com, gI):
        self.x_i += d_com[0][:]
        self.x_i /= gI


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
        self.m_acc_weight = {'OT': 2,
                             'SOT': 2,
                             'CLA': 0,
                             'OC': 0}
        self.rsw = 1.40
        self.dsw = 0.04
        self.x_mcec = np.asarray([0.00, 0.00, 0.00])
        self.correction_groups = []
        self.correction_weights = []

    def calc_mcec(self, rH, rXj, acc_types, correction_groups=None):
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
        self.x_mcec[:] = calc_mcec_location(rH, rXj, self.acc_weights,
                                                 self.rsw, self.dsw)
        # self.x_mcec[:] = self.calc_mcec_location(rH, rXj, self.acc_weights,
        #                                          self.switch, self.rsw, self.dsw)
        print("MCEC before correction", self.x_mcec)
        if correction_groups:
            self.x_mcec[:] += self.calc_mcec_correction(rH, correction_groups)
        print("Final mCEC", self.x_mcec)
    def calc_mcec_correction(self, rH, rGroups):
        num_groups = len(self.correction_groups)
        correction = np.asarray([0.,0.,0.])
        my_correction = np.asarray([0.,0.,0.])
        print('num cor groups, positions',num_groups, rGroups)
        if num_groups != len(rGroups):
            print("Error, the number of groups found does not equal the number of groups parsed")
            raise LookupError
        for g in range(num_groups):
            my_correction[:] = 0.
            group_length = len(self.correction_groups[g])
            group_diff_max = np.empty(group_length)
            for x in range(group_length):
                dists = rH - rGroups[g][x]
                dists = np.linalg.norm(dists, axis=1)
                dists = self.switch(dists, self.rsw, self.dsw)
                #group_diff_max[x] = self.diff_max(dists)
                group_diff_max[x] = dists.max()
            print(group_diff_max)
            for l, k in permutations(range(group_length), 2):
                my_correction += group_diff_max[k] * (rGroups[g][l] - rGroups[g][k])
            my_correction *= self.correction_weights[g]
            correction += my_correction
        print('Correciton amount', correction)
        return correction


    def get_weight_vector(self, types):
        num_acc = len(types)
        self.acc_weights = np.zeros(num_acc, dtype=float)

        for i in range(num_acc):
            try:
                self.acc_weights[i] = self.m_acc_weight[types[i]]
            except LookupError:
                print("Error looking up acceptor %d, type not found" % i)
                raise

    @staticmethod
    def diff_max(results, power=15):
        """
        Differentiable maximum function. Given a list of floats,
        calculate the differentiable maximum funciton
        :param results:
        :return:
        """
        a = results ** power
        b = a * results
        if (a[:] == np.nan).any():
            a[a==np.nan] == 0.0
        if (b[:] == np.nan).any():
            b[b==np.nan] == 0.0
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
    def fos(x):
        """
        Our fifth order spline
        :param x:
        :return:
        """
        return -6*x** 5 + 15*x**4 - 10*x**3 + 1


@jit(parallel=True)
def chakrabarti_switching(d, rsw, dsw):
    """
    Chakrabarti switching function from
    Konig et all
    J. Phys. Chem. A. Vol 110, No.2
    :param d: real distance
    :param rsw: real midpoint of switching function
    :param dsw: real slope of switching function
    :return:
    """
    return (1+np.exp((d-rsw)/dsw))**-1


@jit
def calc_mcec_location(rH, rXj, w, rsw, dsw):
    """
    Returns the mcec location from equation 6 of
    J. Phys. Chem. A, Vol. 110, 2006

    This is the mCEC location without the correction term

    :param rH: np.ndarray of floats with size (n,3) where n is
    number of hydrogens. hydrogen locations
    :param rXj: np.ndarray of floats with size (J,3) where J is
    number of acceptors. acceptor locations
    :param w: np.ndarray of integers with size J representing
    the minimum protonatied state of the acceptor
    :param switch: vectorized switching function that takes a scalar
     distance. Must accept numpy arrays.
    :return: zeta: The mcec without the correction term
    """

    # hydrogen and weighted acceptors
    zeta = np.sum(rH, axis=0)
    print('sum_hydrogen', zeta)
    zeta -= np.dot(w, rXj)
    print('after subtracting acc', zeta)
    num_m = rH.shape[0]
    num_j = rXj.shape[0]
    slow = False # Is 2 seconds slower for 25 second job...
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
        zeta -= np.tensordot(rHXj, chakrabarti_switching(np.linalg.norm(rHXj, axis=2), rsw, dsw), [(0,1),(0,1)])
    return zeta