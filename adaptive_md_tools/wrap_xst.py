#!/usr/bin/env python
"""
Stand alone program that wraps structure according to xst file
"""
__author__ = 'Adam Duster'
__copyright__ = ''
__credits__ = ['Adam Duster']
__license__ = 'CC-BY-SA'
__version__ = '0.1'
__email__ = 'adam.duster@ucdenver.edu'
__status__ = 'Development'

import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations.translate import center_in_box
#from MDAnalysis.lib._cz

def get_args(args=None):
    """ This is written as a default function to put at beginning of all Python
    scripts which require command line arguments. This uses the argparse module
    which must be declared in the main program to ensure that the object is able
    to be used by the caller
    --Adam Duster 21 June 2017
    """
    parser = argparse.ArgumentParser(description='see header of python script')
    parser.add_argument(
        '-i',
        '--input',
        help='Input file coordinates',
        required=True)
    parser.add_argument(
        '-s',
        '--structure',
        help='Input file structure',
        required=True,
        default=0)
    parser.add_argument(
        '-x',
        '--xst',
        help='XST file with periodic boundaries (1 line per frame in input)',
        required=True,
        default=False)
    parser.add_argument(
        '-o',
        '--out',
        help='Path to write coordinates to',
        required=True,
        default=False)
    # parser.add_argument(
    #     '-c',
    #     '--center',
    #     help='Center of box. If one int is given, then that atom will be center. If 3 floats given, then that will be center of the box. Otherwise it will be [0, 0, 0]',
    #     required=False,
    #     nargs='+',
    #     default=None
    # )
    parser.add_argument(
        '-c',
        '--center',
        help='Atom to center box on',
        required=False,
        type=int,
        default=None
    )
    return parser.parse_args(args)

def read_xst_triclinic(xst_path):
    data = np.loadtxt(xst_path)
    steps = data[:,0].astype(int)
    a = data[:, 1:4]
    b = data[:, 4:7]
    c = data[:, 7:10]
    o = data[:, 10:13]
    return steps, a, b, c, o

def read_xst_cube(xst_path):
    steps, x, y, z = np.loadtxt(xst_path, usecols=[0, 1, 5, 9])


def get_center(arg, pos):
    if arg == None:
        return [0, 0, 0]
    elif len(arg) == 3:
        return arg
    elif len(arg) == 1:
        ind = int(arg)
        return pos[ind]
    else:
        print('Fatal error, check argument for center.')
        raise


def main():
    ## Vars
    arg_vals = [
        '-i', '/ml/water_in_water/behler_h2o/training-set/dataset_ang.xyz',
        '-s', '/ml/water_in_water/behler_h2o/training-set/inp/one.psf',
        '-x', '/ml/water_in_water/behler_h2o/training-set/inp/one.xst',
        '-o', '/ml/water_in_water/behler_h2o/training-set/out.xyz',
        '-c', 0
    ]
    args = get_args(arg_vals)

    # Load files
    u = mda.Universe(args.structure, args.input)
    #steps, x, y, z = np.loadtxt(args.xst, usecols=[0, 1, 5, 9])
    steps, a, b, c, o = read_xst_triclinic(args.xst)
    sel = u.select_atoms('all')
    out = mda.Writer(args.out, sel.n_atoms)

    #XST file and coordinates must have same # steps
    assert len(steps) == len(u.trajectory)

    for s in steps:
        ts = u.trajectory[s]
        ts.triclinic_dimensions = [a[s], b[s], c[s]]

        if args.center != None:
            ag = u.select_atoms("bynum %d" % (args.center + 1))
            ts = center_in_box(ag, center='mass')(ts)
        sel.wrap(compound='atoms', center='com')


        # # Get the center of the box
        # if args.center != None:
        #     ag = u.select_atoms("bynum %d" % (args.center + 1))
        #     ts = center_in_box(ag, center='mass')(ts)
        # mda.transformations.wrap()
        # #sel.wrap(compound='atoms', center='com')
        # #sel.unwrap(compound='atoms', center='com')
        out.write(sel)

    out.close()


if __name__ == '__main__':
    main()