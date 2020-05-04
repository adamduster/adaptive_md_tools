#!/usr/bin/env python
"""
This is currently a set of classes that contains the bonding
topology and AP groups for the system
"""
__author__ = 'Adam Duster'
__copyright__ = ''
__credits__ = ['Adam Duster']
__license__ = 'CC-BY-SA'
__version__ = '0.1'
__email__ = 'adam.duster@ucdenver.edu'
__status__ = 'Development'

import argparse
import sys
import numpy as np
from itertools import (takewhile,repeat)


__el2num = {'H':1,
            'O':8,
            'C':6,
            'N':7}
__num2el = {1:'H',
            8:'O',
            6:'C',
            7:'N'}


class Bonds:
    """
    A class to keep track of the bonding for system
    """
    def __init__(self, numAtoms, bond_list=[]):
        self.numAtoms = numAtoms
        self.bonds = [[] for x in range(numAtoms)]
        self.numBonds = 0
        self.blist = []

        for b in bond_list:
            self.addBonds(b[0], b[1])
        return

    def addBonds(self, a1, a2):
        """
        Add a bond to a1 and a2. This means a1 is added to a2's bonds and
        a2 is added to a1's bonds
        :param a1:
        :param a2:
        :return:
        """
        if a2 in self.bonds[a1]:
            print("Cannot add bond, atom %d is already bonded to atom %d"
                  % (a1, a2))
            raise RuntimeError

        if a1 in self.bonds[a2]:
            print("Cannot add bond, atom %d is already bonded to atom %d"
                  % (a2, a1))
            raise RuntimeError

        if a1 == a2:
            print("Cannot bond atom to itself")
            raise RuntimeError
        self.bonds[a1].append(a2)
        self.bonds[a2].append(a1)
        self.numBonds += 1
        self.blist.append((a1,a2))
        return

    def printBonds(self, style=None):
        pBonds = self.getUniqueBonds()
        nBonds = len(pBonds)
        bLine = "{0:9d}   {1:9d}"
        psfLine = "{0:8d}{1:8d}{2:8d}{3:8d}{4:8d}{5:8d}{6:8d}{7:8d}"
        partLine = "{0:8d}{1:8d}"
        if not style:
            print("NBonds: " + str(nBonds))
            for i in range(nBonds):
                print(bLine.format(*pBonds[i]))
        if style == 'psf':
            tBonds = []
            for i in pBonds:
                tBonds.append((i[0]+1, i[1]+1))
            pBonds = tBonds
            print("{0:8d} !NBONDS: bonds".format(nBonds))
            nLines = nBonds // 4
            b = 0
            for i in range(nLines):
                print(psfLine.format(*pBonds[b], *pBonds[b+1], *pBonds[b+2],
                                     *pBonds[b+3]))
                b += 4
            if nBonds % 4 > 0:
                lastLine = ""
                for i in range(nBonds % 4):
                    lastLine += partLine.format(*pBonds[b])
                    b += 1
                print(lastLine)
        return

    def getUniqueBonds(self):
        uniqueBonds = []
        for i in range(self.numAtoms - 1):
            for j in self.bonds[i]:
                if j > i:
                    uniqueBonds.append((i, j))
        return uniqueBonds


class Groups:
    """
    A class to keep track of the AP group stuff
    """
    def __init__(self, numGroups, numAtoms):
        self.numAtoms = numAtoms
        self.numGroups = numGroups
        self.atomGroups = np.full(numAtoms, -1, dtype=int)
        self.groupAtoms = [[] for x in range(numGroups)]
        self.groupReps = np.full(numGroups, -1, dtype=int)
        self.nonDefaultRepresentative = False

    def setAtomGroup(self, ind, group):
        """
        Affiliate the atom with a group. If this is the first atom
        affiliated with the group, make it the group representative.
        :param ind: the atom index
        :param group: the group to affiliate atom with
        :return:
        """
        if ind > self.numAtoms - 1:
            print("Error, atom index out of range for setting group")
            raise IndexError
        if group > self.numGroups - 1:
            print("Error, group index out of range for setting group")
        self.atomGroups[ind] = group
        if ind not in self.groupAtoms[group]:
            self.groupAtoms[group].append(ind)
            self.groupReps[group] = self.groupAtoms[group][0]
        else:
            print("Error, atom %d already in group %d" % (ind, group))
            raise RuntimeError

        return

    def setGroupRep(self, ind, group):
        """
        Set the group representative
        :param ind: atom index to make representative
        :param group: group to make representative of
        :return:
        """
        if ind in self.groupAtoms[group]:
            self.groupAtoms[group].insert(0, self.groupAtoms[group].pop(
                self.groupAtoms[group].index(ind)))
            self.groupReps[group] = self.groupAtoms[group][0]
        else:
            "Atom %d cannot be group representative because it is not in the"
            " group!"
            raise RuntimeError
        return

    def checkGroups(self):
        """
        Check to ensure all atoms are assigned to group and all groups have
        a representative atom. Return false if this is not the case
        :return: goodCheck = logical
        """
        goodCheck = True
        # Check that groups all have a representative
        for i in range(self.numGroups):
            if not self.groupAtoms[i]:
                print("Group %d no atoms" % i)
                goodCheck = False

        # Check that all atoms are assigned to group
        for i in range(self.numAtoms):
            if self.atomGroups[i] < 0:
                print("Atom %d not assigned to group" % i)
                goodCheck = False
        return goodCheck

    def printGroups(self, style='groups.log'):
        header = '# NGroups       NAtoms'
        header2 = '# Group       AtomIndex (both 1 based)'
        outLine = '{0:9d}    {1:9d}'
        if style == 'groups.log':
            print(header)
            print(outLine.format(self.numGroups, self.numAtoms))
            print(header2)
            for i in range(self.numGroups):
                for j in range(len(self.groupAtoms[i])):
                    print(outLine.format(i + 1, self.groupAtoms[i][j] + 1))
            print('END')

    def transferAtom(self, atom, new_atom):
        """
        Change the group of an atom specified by ATOM to the group of the atom
        specified by NEW_BOND
        :param atom:
        :param new_atom:
        :return:
        """
        old_group = self.atomGroups[atom]
        new_group = self.atomGroups[new_atom]
        self.groupAtoms[old_group].remove(atom)
        self.groupAtoms[new_group].append(atom)
        self.atomGroups[atom] = new_group
        return


class Topology:
    def __init__(self, of_path=None, num_atoms=0):
        self.hello = True
        self.bonds = []
        self.psf_lines = []
        self.num_atoms = 0
        self.ofi = None
        if of_path:
            self.set_ofi(of_path)
        self.mm_types = MMTypes(1)
        # if num_atoms:
            # self.ind = np.arange(1, num_atoms + 1, 1)
            # self.segment = np.array(num_atoms, dtype='U4')
            # self.resid = np.array(num_atoms, dtype=int)
            # self.resname = np.array(num_atoms, dtype='U4')
            # self.atmname = np.array(num_atoms, dtype='U4')
        # self.ind = []
        # self.resid = []
        # self.segname = []
        # self.resname = []
        # self.atmname = []
        # self.atmtype = []
        # self.charge = []
        # self.mass = []


    def set_ofi(self, of_path):
        if self.ofi:
            self.ofi.close()
        else:
            self.ofi = open(of_path, 'w')

    def close_ofi(self):
        self.ofi.close()
        self.ofi = None

    def write_psf(self, bonds, num_atoms, start_lines=None):
        num_bonds = len(bonds)
        num_ang = 0
        num_dihed = 0
        num_imp = 0
        num_don = 0
        num_acc = 0
        num_nnb = 0
        bond_tstr = "{0:8d} !NBOND: bonds\n"
        ang_tstr = "{0:8d} !NTHETA: angles\n"
        dihed_tstr = "{0:8d} !NPHI: angles\n"
        imp_tstr = "{0:8d} !NIMPHI: angles\n"
        don_tstr = "{0:8d} !NDON: angles\n"
        acc_tstr = "{0:8d} !NACC: angles\n"
        nnb_tstr = "{0:8d} !NNB: angles\n\n"
        ngrp = "{:8d}{:8d} !NGRP\n{:8d}{:8d}{:8d}\n".format(1,0,0,0,0)

        if start_lines:
            self.ofi.writelines(start_lines)
        else:
            self.write_title()
            self.write_natom()
        self.write_section(bond_tstr, bonds, 4)
        # Write angles
        self.ofi.write(ang_tstr.format(num_ang))
        self.ofi.write('\n\n')
        self.ofi.write(dihed_tstr.format(num_dihed))
        self.ofi.write('\n\n')
        self.ofi.write(imp_tstr.format(num_imp))
        self.ofi.write('\n\n')
        self.ofi.write(don_tstr.format(num_don))
        self.ofi.write('\n\n')
        self.ofi.write(acc_tstr.format(num_acc))
        self.ofi.write('\n\n')
        self.ofi.write(nnb_tstr.format(num_nnb))
        self.ofi.write('\n\n')
        zeros = [[-1] for i in range(num_atoms)]
        self.write_section(nnb_tstr, zeros, 8)
        self.ofi.write(ngrp)
        self.close_ofi()

    def write_title(self):
        """
        Write a title for the psf file
        :return:
        """
        self.ofi.write('PSF\n\n')
        self.ofi.write('{0:8d} !NTITLE\n'.format(1))
        self.ofi.write(' REMARKS nothing\n\n')

    def write_natom(self):
        """
        Write the mm atom types into the psf file
        :return:
        """
        line = ""
        self.ofi.write('{:8d} !NATOM\n'.format(self.mm_types.numAtoms))
        for i in range(self.mm_types.numAtoms):
            self.ofi.write("{:8d} "    .format(i+1))
            self.ofi.write("{:4s} "    .format(self.mm_types.segName[i]))
            self.ofi.write("{:<4d} "   .format(self.mm_types.resId[i]))
            self.ofi.write("{:4} "     .format(self.mm_types.resName[i]))
            self.ofi.write("{:4} "     .format(str(self.mm_types.atomName[i])))
            self.ofi.write("{:4} "     .format(str(self.mm_types.atomType[i])))
            self.ofi.write("{:10.6f}  ".format(self.mm_types.charge[i]))
            self.ofi.write("{:12.4f}"  .format(self.mm_types.atomMass[i]))
            self.ofi.write("{:12d}\n"  .format(0))
        self.ofi.write('\n')
        return

    def write_section(self, tstring, inds, groups_per_line):
        num_groups = len(inds)
        num_lines = num_groups // groups_per_line
        extra_bonds = num_groups % groups_per_line

        self.ofi.write(tstring.format(num_groups))
        g = 0
        gpl = groups_per_line
        for out_line in range(num_lines):
            my_str = self.make_str(inds[g:g+gpl])
            self.ofi.write(my_str)
            g += gpl
        if extra_bonds:
            my_str = self.make_str(inds[g:])
            self.ofi.write(my_str)
        self.ofi.write('\n\n')


    def make_str(self, inds, str_width=8):
        template = "{:" + str(str_width) + "d}"
        my_str = ""
        num_groups = len(inds)
        num_in_group = len(inds[0])
        for i in range(num_groups):
            for j in range(num_in_group):
                my_str += template
        my_str += '\n'
        flat_list = [item +1 for sublist in inds for item in sublist]
        return my_str.format(*flat_list)

    def write_mol2(self, bonds):


        self.ofi.write('@<TRIPOS>MOLECULE\n')
        self.ofi.write('*****\n')
        s = ' {0:d} {1:d} 0 0 0\n'
        self.ofi.write(s.format(self.num_atoms, len(bonds)))
        self.ofi.write('SMALL\nGASTEIGER\n\n')
        self.write_atom()
        self.write_bond(bonds)
        self.close_ofi()

    def write_atom(self):
        self.ofi.write('@<TRIPOS>ATOM\n')
        atmstr = '{0:7d} {1:2}          0.0000    0.0000    0.0000 {2:4} {5:>4}  {3:5}       {4:7.4f}\n'
        for i in range(self.mm_types.numAtoms):
            self.ofi.write(atmstr.format(i+1,
                                         self.mm_types.element[i],
                                         str(self.mm_types.atomType[i]),
                                         str(self.mm_types.atomName[i]),
                                         self.mm_types.charge[i],
                                         str(self.mm_types.resId[i])))
    def write_bond(self, bonds):
        self.ofi.write('@<TRIPOS>BOND\n')
        s = '{:6d}{:6d}{:6d}{:6d}\n'
        for i in range(len(bonds)):
            self.ofi.write(s.format(i+1, bonds[i][0]+1, bonds[i][1]+1, 1))


class MMTypes:
    def __init__(self, numAtoms):
        self.numAtoms = numAtoms
        self.strWidth = 4
        sid = "U%d" % self.strWidth
        self.segName = np.zeros(numAtoms, dtype=sid)
        self.resId = np.zeros(numAtoms, dtype=int)
        self.resName = np.zeros(numAtoms, dtype=sid)
        self.charge = np.zeros(numAtoms, dtype=float)
        self.atomName = np.zeros(numAtoms, dtype=sid)
        self.atomType = np.zeros(numAtoms, dtype=sid)
        self.atomMass = np.zeros(numAtoms, dtype=float)
        self.element = np.zeros(numAtoms, dtype='U2')


class File:
    def __init__(self, filePath, fileType=''):
        try:
            self.ifi = open(filePath, 'r')
            self.path = filePath
        except FileNotFoundError:
            print("Cannot open {0} file: ".format(fileType) + filePath)
            sys.exit()


class Mol2File(File):
    def getTopo(self):
        topo = Topology()
        topo.mm_types = self.getMMTypes()
        topo.num_atoms = topo.mm_types.numAtoms
        topo.bonds = self.getBonds()
        return topo

    def getNumAtoms(self):
        self.ifi.seek(0)
        while True:
            line = self.ifi.readline()
            words = line.split()
            if 'MOLECULE' in line:
                line = self.ifi.readline()
                words = self.ifi.readline().split()
                return int(words[0]), int(words[1])
        print("Could not find number of atoms")
        raise ValueError

    def getMMTypes(self):
        numAtoms, numBonds = self.getNumAtoms()
        mmNames = MMTypes(numAtoms)
        self.ifi.seek(0)
        found_atoms = False
        while True:
            line = self.ifi.readline()
            if not line:
                break
            if 'ATOM' in line.upper() and not found_atoms:
                self.read_atom(mmNames)
                found_atoms = True
        if not found_atoms:
            print("Error, could not find ATOM section")
            raise LookupError
        return mmNames


    def getBonds(self):
        numAtoms, numBonds = self.getNumAtoms()
        bonds = Bonds(numAtoms)
        self.ifi.seek(0)
        found_bonds = False
        while True:
            line = self.ifi.readline()
            if not line:
                break
            if 'BOND' in line.upper() and not found_bonds:
                self.read_bonds(bonds, numBonds)
                found_bonds = True
        if not found_bonds:
            print("Error, could not find bonds")
            raise LookupError
        return bonds

    def read_atom(self, mmType):
        for i in range(mmType.numAtoms):
            line = self.ifi.readline()
            words = line.split()
            mmType.element[i] = words[1]
            mmType.atomType[i] = words[5]
            mmType.atomName[i] = words[7]
            mmType.charge[i] = float(words[8])

    def read_bonds(self, bonds, numBonds):
        for i in range(numBonds):
            line = self.ifi.readline()
            words = line.split()
            bonds.addBonds(int(words[1])-1, int(words[2])-1)


class PDBFile(File):
    def __init__(self, pdbPath, required=None):
        File.__init__(self, pdbPath, 'pdb')
        self.required = required  # required fields to parse
        self.numAtoms = 0

    def parsePDB(self):
        serial = []
        name = []
        altLoc = []
        resName = []
        chainID = []
        resSeq = []
        iCode = []
        x = []
        y = []
        z = []
        occupancy, tempFactor, element, charge = [], [], [], []

        while True:
            line = self.ifi.readline()
            if not line:
                break
            if len(line) < 6: continue
            if ('ATOM' in line[:6].upper()) or ('HETATM' in line[:6].upper()):
                break
        while line.lstrip()[:3].upper() != 'END':
            try:
                serial.append(int(line[6:11]))
            except:
                pass
            try:
                name.append(line[12:16])
            except:
                pass
            try:
                altLoc.append(line[16])
            except:
                pass
            try:
                resName.append(line[17:20])
            except:
                pass
            try:
                chainID.append(line[21])
            except:
                pass
            try:
                resSeq.append(line[22:26])
            except:
                pass
            try:
                iCode.append(line[26])
            except:
                pass
            try:
                x.append(float(line[30:38]))
            except:
                pass
            try:
                y.append(float(line[38:46]))
            except:
                pass
            try:
                z.append(float[line[46:54]])
            except:
                pass
            try:
                occupancy.append(float(line[54:60]))
            except:
                pass
            try:
                tempFactor.append(float(line[60:66]))
            except:
                pass
            try:
                element.append(line[76:78].strip())
            except:
                if 'element' in self.required:
                    print("Error, could not parse element from pdb "+ self.path)
                    sys.exit()
            try:
                charge.append(line[78:80])
            except:
                pass

            self.numAtoms += 1
            line = self.ifi.readline()

        element = np.asarray(element)
        return element


class PSFFile:
    def __init__(self, psfPath):
        try:
            self.ifi = open(psfPath, 'r')
        except FileNotFoundError:
            print("Cannot find psf file: " + psfPath)
            sys.exit()

    def getNumAtoms(self):
        """
        Read the number of atoms from the PSF file
        This leave the file at the NATOM line! Rewind it if you need to
        go from beginning and don't change because other programs depend
        on it!
        :return:
        """
        self.ifi.seek(0)
        natoms = 0
        while True:
            line = self.ifi.readline()
            if not line:
                break
            if "!NATOM" in line:
                try:
                    natoms = int(line.split()[0])
                    break
                except IOError:
                    print("Cannot read natoms from psf file")
                    sys.exit()
        if natoms == 0:
            print("Could not find NATOM in psf file")
            raise IOError

        return natoms

    def getTopo(self):
        topo = Topology()
        topo.mm_types = self.getMMTypes()
        topo.num_atoms = topo.mm_types.numAtoms
        return topo


    def getBonds(self):
        numAtoms = self.getNumAtoms()
        bonds = Bonds(numAtoms)
        self.ifi.seek(0)
        foundNBond = False
        while True:
            line = self.ifi.readline()
            if not line:
                break
            if "!NBOND" in line:
                foundNBond = True
                break

        if not foundNBond:
            print("Could not find NBonds section in psf file")
            raise IOError

        try:
            nbonds = int(line.split()[0])
        except IOError:
            print("Cannot read Nbonds")
            sys.exit()

        bondLines = nbonds // 4
        if nbonds % 4 > 0:
            bondLines += 1

        foundBonds = 0

        for l in range(bondLines):
            line = self.ifi.readline()
            words = line.split()
            nwords = len(words)
            nowBonds = nwords // 2
            for i in range(nowBonds):
                a1 = int(words[i * 2]) - 1
                a2 = int(words[i * 2 + 1]) - 1
                bonds.addBonds(a1, a2)
                foundBonds += 1

        if foundBonds != nbonds:
            print("Did not find all of the bonds!")
            raise RuntimeError

        return bonds

    def read_beginning_lines(self):
            lines = []
            self.ifi.seek(0)
            line = ""
            found = False
            while found == False:
                line = self.ifi.readline()
                if "!NBOND" not in line:
                    lines.append(line)
                else:
                    found = True
            if not found:
                print("Error parsing PSF file for lines")
            return lines


    def getMMTypes(self):
        numAtoms = self.getNumAtoms()
        mmNames = MMTypes(numAtoms)
        for i in range(numAtoms):
            try:
                line = self.ifi.readline()
                words = line.split()
            except IOError:
                print("Error reading line %d in psf file" % i)
                sys.exit('mmtypes 1')

            if len(words) != 9:
                print("Error, not enough data in psf file for atom %d " % i)
                sys.exit('mmtypes 2')

            try:
                mmNames.segName[i] = words[1]
            except IOError:
                print("Error reading segname")
                sys.exit('mmtypes 3')
            try:
                mmNames.resId[i] = int(words[2])
            except IOError:
                print("Error reading residue index")
                sys.exit('mmtypes 4')
            try:
                mmNames.resName[i] = words[3]
            except IOError:
                print("Error reading resdue name")
                sys.exit('mmtypes 5')
            try:
                mmNames.atomName[i] = words[4]
            except IOError:
                print("Error reading atom name")
                sys.exit('mmtypes 6')
            try:
                mmNames.atomType[i] = words[5]
            except IOError:
                print("Error reading atom type")
                sys.exit('mmtypes 7')
            try:
                mmNames.charge[i] = float(words[6])
            except IOError:
                print("Error reading charge")
                sys.exit('mmtypes 8')
            try:
                mmNames.atomMass[i] = float(words[7])
            except IOError:
                print("Error reading atom mass")
                sys.exit('mmtypes 9')

            except IOError:
                print("Error reading psf at atom %d" % i)
                sys.exit()
        return mmNames


class GroupsFile:
    def __init__(self, groupsPath):
        try:
            self.ifi = open(groupsPath, 'r')
            self.groupsPath = groupsPath
        except FileNotFoundError:
            print("Cannot find groups file: " + groupsPath)
            sys.exit()

    def getGroups(self):

        # Read the number of groups, then number of atoms from the groups file
        try:
            line = _rLine(self.ifi)
            numGroups, numAtoms = [int(i) for i in line.split()[:2]]
        except IOError:
            print("Error reading number of groups and atoms from groups file: "
                  + self.groupsPath)
            sys.exit()

        # Initialize the groups structure
        groups = Groups(numGroups, numAtoms)

        # Fill up the groups
        while True:
            line = _rLine(self.ifi)
            if not line:
                print('Error reading groups file, "END" not encountered')
                raise IOError
            words = line.split()
            if words[0].upper() == 'END':
                break

            group = int(words[0]) - 1
            ind = int(words[1]) - 1
            groups.setAtomGroup(ind, group)

        # Exit if there is anything left over
        if not groups.checkGroups():
            print("Error, not all groups or atoms were in groups file")
            raise RuntimeError

        return groups


def _rLine(ifi):
    """
    By Adam Duster, Oct 12, 2017
    Read a line. Skip the line if there is a comment sign (#). If the line
    fails to read, raise an exception to be handeled by the calling program
    """
    while True:
        line = ifi.readline()
        if not line:
            raise EOFError

        # skip comment
        if line.strip().startswith('#'):
            continue
        else:
            return line


def get_args(args=None):
    """ This is written as a default funtion to put at beginning of all Python
    scripts which require command line arguments. This uses the argparse module
    which must be declared in the main program to ensure that the object is able
    to be used by the caller
    --Adam Duster 21 June 2017
    """
    parser = argparse.ArgumentParser(description='see header of python script')
    parser.add_argument(
        '-i',
        '--input',
        help='Input file name',
        required=True)
    parser.add_argument(
        '-v',
        '--verbose',
        help='Controls the level of output, use multipe v for more output',
        required=False,
        action='count',
        default=0)
    parser.add_argument(
        '-c',
        '--center',
        help='atom index to be moved to center',
        required=True,
        type=int
    )
    parser.add_argument(
        '-x',
        '--x',
        help='x length',
        required=True,
        type=float
    )
    parser.add_argument(
        '-y',
        '--y',
        help='y length',
        required=True,
        type=float
    )
    parser.add_argument(
        '-z',
        '--z',
        help='z length',
        required=True,
        type=float
    )
    parser.add_argument(
        '-d',
        '--debug',
        help='Enter debug mode',
        required=False,
        action='store_true',
        default=False)
    parser.add_argument(
        '--psf',
        help="PSF file path for reading bonds",
        required=False,
        default=None
    )
    return parser.parse_args(args)


def read_xyz(ifpath, return_coord_array=False):
    """
    Reads an xyz file
    :param ifpath:
    :return: num_atoms, x, y, z, atom_type, title as numpy int or float arrays
    """
    with open(ifpath) as ifi:
        try:
            num_atoms = int(ifi.readline())
        except RuntimeError:
            raise RuntimeError("Error reading n atoms")

        title = ifi.readline()
        x = np.zeros(num_atoms, dtype=float)
        y = np.zeros(num_atoms, dtype=float)
        z = np.zeros(num_atoms, dtype=float)
        atom_type = [''] * num_atoms
        for i in range(num_atoms):
            line = ifi.readline()
            words = line.split()
            atom_type[i] = words[0]
            x[i] = float(words[1])
            y[i] = float(words[2])
            z[i] = float(words[3])
            if atom_type[i] == 'OH2':
                atom_type[i] = 'O'
            elif atom_type[i] == 'H1' or atom_type[i] == 'H2'\
                    or atom_type[i] == 'H3':
                atom_type[i] = 'H'
    if return_coord_array:
        return num_atoms, np.stack((x,y,z), axis=-1), atom_type, title
    return num_atoms, x, y, z, atom_type, title


def read_xyz2(ifpath):
    """
    Reads an xyz file and returns numpy array and atomic numbers from element
    :param ifpath:
    :return: num_atoms, coords, atom_type, atom_number as numpy int or float
    :rtype: (np.ndarray(n, dtype=int16), np.ndarray(n,3, dtype=float),
    np.ndarray(n, dtype="U2"), np.ndarray(n, dtype=int8)
    """
    el_dict = {'H':1, 'C':6, 'N':7, 'O':8}
    num_dict = {1:'H', 6:'C', 7:'N', 8:'O'}
    with open(ifpath) as ifi:
        try:
            num_atoms = np.int16(ifi.readline())
        except:
            sys.exit("Error reading n atoms")
        coords = np.zeros((num_atoms,3), dtype=float)
        atom_type = np.zeros(num_atoms, dtype="U2")
        atomic_number = np.zeros(num_atoms, dtype=np.int8)
        ifi.readline()
        for i in range(num_atoms):
            line = ifi.readline()
            words = line.split()
            atom_type[i] = words[0]
            coords[i, 0] = float(words[1])
            coords[i, 1] = float(words[2])
            coords[i, 2] = float(words[3])
        if atom_type[i][0].isdigit():
            for i, el in enumerate(atom_type):
                atomic_number[i] = int(atom_type[i])
                atom_type[i] = num_dict[atomic_number[i]]

    return num_atoms, coords, atom_type


def read_xyz_trajectory(ifpath):
    """
    Reads an xyz file
    :param ifpath:
    :return: num_atoms, x, y, z, atom_type, title as numpy int or float arrays
    """
    with open(ifpath) as ifi:
        try:
            num_atoms = int(ifi.readline())
        except RuntimeError:
            raise RuntimeError("Error reading n atoms")
        nlines = rawincount(ifpath)
        if nlines % (num_atoms + 2) != 0:
            raise RuntimeError("Improper number of lines in trajectory")
        nframes = nlines//(num_atoms+2)

        title = ifi.readline()
        coords = np.zeros((nframes, num_atoms, 3), dtype=float)
        atom_type = [''] * num_atoms
        ifi.seek(0)
        for f in range(nframes):
            line = ifi.readline()
            line = ifi.readline()
            for i in range(num_atoms):
                line = ifi.readline()
                words = line.split()
                try:
                    atom_type[i] = words[0]
                    coords[f, i, 0] = float(words[1])
                    coords[f, i, 1] = float(words[2])
                    coords[f, i, 2] = float(words[3])
                except:
                    print("Error reading xyz trajectory at frame %d" % f)
                    print("Current line:")
                    print(line)
                    sys.exit()
                if atom_type[i] == 'OH2':
                    atom_type[i] = 'O'
                elif atom_type[i] == 'H1' or atom_type[i] == 'H2'\
                        or atom_type[i] == 'H3':
                    atom_type[i] = 'H'
    return num_atoms, coords, atom_type, title


def rawincount(filename):
    """
    Get the number of lines in filename
    :param filename:
    :return:
    """
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )


def wrap_coords(cell_x, cell_y, cell_z, x, y, z, num_atoms, center):
    """
    1. Translate all atoms by vector from atom designated by center
    to center of box defined by [[0, cell_x],[0, cell_y], [0, cell_z]]
    2. wrap all atoms outside of the above box into the box

    :param cell_x: x cell dimention float
    :param cell_y: y cell dimention float
    :param cell_z: float z cell dimension
    :param x: np float array x coords
    :param y: np float array y coords
    :param z: np float array z coords
    :param num_atoms: int
    :param center: int 0-based index of atom center
    :return: x, y, z
    """
    def wrap_dimension(x, cell_x):
        no_wraps = False
        while not no_wraps:
            no_wraps = True
            for i in range(x.size):
                if x[i] < 0:
                    x[i] += cell_x
                    no_wraps = False
                elif x[i] > cell_x:
                    x[i] -= cell_x
                    no_wraps = False
        return x



    # Find the translation vector to shift all atoms by the vector
    # from the desired centers coords to the new center
    center_a_coords = np.asarray([x[center], y[center], z[center]])
    new_center = np.asarray([cell_x/2., cell_y/2., cell_z/2.])
    translation = new_center - center_a_coords

    x[:] += translation[0]
    y[:] += translation[1]
    z[:] += translation[2]


    # Now wrap the atoms
    x = wrap_dimension(x, cell_x)
    y = wrap_dimension(y, cell_y)
    z = wrap_dimension(z, cell_z)
    return x, y, z


def print_xyz(num_atoms, x, y, z, atom_type, title=None, ofi=None):
    """
    Print out a geometry with 4 float columns and the atom type

    Parameters
    ----------
    num_atoms: int
        The number of atoms in the geometry
    x: list of floats
        The x coordinates of the atom
    y: list of floats
        The y coordinates of the atom
    z: list of floats
        The z coordinates of the atom
    atom_type: list of int or str
        The elmental symbol or atomic number for the atom
    title: string, optional
        The title line for the xyz file
    ofi: _io.TextIOWrapper
        Optional location to print the xyz (default is sys.stdout)

    Returns
    -------
        None
    """

    print(num_atoms, file=ofi)
    if title:
        print(title[:-1], file=ofi)
    else:
        print("", file=ofi)
    l = "{0:2}    {1:10.6f}    {2:10.6f}    {3:10.6f}"
    for i in range(num_atoms):
        print(l.format(atom_type[i], x[i], y[i], z[i]), file=ofi)


def print_sispa(num_atoms, x, y, z, pi, atom_type, title=None, ofi=None):
    """
    Print out a geometry with 4 float columns and the atom type

    Parameters
    ----------
    num_atoms: int
        The number of atoms in the geometry
    x: list of floats
        The x coordinates of the atom
    y: list of floats
        The y coordinates of the atom
    z: list of floats
        The z coordinates of the atom
    pi: list of floats
        The pi (or other float) value for the atom
    atom_type: list of int or str
        The elmental symbol or atomic number for the atom
    title: string, optional
        The title line for the xyz file
    ofi: _io.TextIOWrapper
        Optional location to print the xyz (default is sys.stdout)

    Returns
    -------
        None
    """
    print(num_atoms, file=ofi)
    if title:
        print(title[:-1], file=ofi)
    else:
        print("", file=ofi)
    l = "{0:2}    {1:10.6f}    {2:10.6f}    {3:10.6f}           {4:10e}"
    for i in range(num_atoms):
        print(l.format(atom_type[i], x[i], y[i], z[i], pi[i]), file=ofi)


def print_xyz_traj(coords, types, title=None, ofi=None):
    """
    Write an xyz trajectory file to ofi
    :param coords: float ndarray with size [f,num_atoms,3] where f is num frames
    :param types: elements
    :param title: title for each frame
    :param ofi: open file handle for writing
    :return:
    """
    num_atoms = coords.shape[1]
    for f in range(coords.shape[0]):
        print_xyz(num_atoms, coords[f,:,0], coords[f,:,1], coords[f,:,2],
                  types, title=title, ofi=ofi)


def get_elements(in_path, file_type):

    if file_type == 'xyz':
        _,_,_,_,elements,_ = read_xyz(in_path)
    elif file_type == 'pdb':
        pdbFile = PDBFile(in_path)
        elements = pdbFile.parsePDB()
    return np.asarray(elements)


def testPSFBonds(psfPath='./test_files/h3o.psf'):
    psf = PSFFile(psfPath)
    bonds = psf.getBonds()
    bonds.printBonds(style='psf')


def testGroups(groupsPath='./test_files/groups.log'):
    gFile = GroupsFile(groupsPath)
    groups = gFile.getGroups()
    groups.printGroups()

# ifpath = args.input
# cell_x = args.x
# cell_y = args.y
# cell_z = args.z
# center = args.center
# verbose = args.verbose
# debug = args.debug
#
# num_atoms, x, y, z, atom_type, title = read_xyz(ifpath)
# x, y, z = wrap_coords(cell_x, cell_y, cell_z, x, y, z, num_atoms, center)
# print_xyz(num_atoms, x, y, z, atom_type, title)


def main():
    arg_vals = None
    args = get_args(arg_vals)


if __name__ == "__main__":
    testGroups()



