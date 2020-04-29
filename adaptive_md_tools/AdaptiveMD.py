#!/usr/bin/env python
"""
This is the file that contains the main program.

This code offers many features that are useful for analyzing adaptive
partitioning simulations.

Translates the system such that the donor
is in the center. Then calculates the indicator and retranslates the system
such that the indicator is now in the center. Also keeps track of the
topology and groups for the system.
"""
import os
import argparse
from adaptive_md_tools.indicator import *
from adaptive_md_tools.core_loop import core_loop


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
        '-d',
        '--debug',
        help='Enter debug mode',
        required=False,
        action='store_true',
        default=False)
    return parser.parse_args(args)


def read_input_file(ifpath):
    """
    Read the input file and save the parameters

    Parameters
    ----------
    ifpath: str
        input file path

    Returns
    -------
    keywords: dict
        Dictionary of parameters
    """
    try:
        ifi = open(ifpath, 'r')
    except FileNotFoundError:
        print("Error opening input file: " + ifpath)
        sys.exit()
    keywords = {"groups_path": None,
                "rlist": 3.5,
                "indicator_verbose": False,
                "proton_types": [],
                "write_freq": 20,
                "write_folder": './',
                "write_partitions": False,
                "active_radius": 5.0,
                "buffer_radius": 0.6,
                "pap_order": 2,
                "write_prefix": '',
                "allow_hop": True,
                "dcd_pbc": False,
                "verbose": False,
                "ind_output_freq": 0,
                "wrap": True,
                "wrap_style": 'fragments',
                "dimensions": [0, 0, 0],
                "ind_method": -1,
                "topology": None,
                "topology_type": 'psf',
                "rdh0": [],
                "ratio_topology_change": True,
                "mcec": False,
                "rsw": 1.4,
                "dsw": 0.04,
                'mcec_g': [],
                'out_coords': None,
                'ap_write_probability': None
                }
    print("*********************** INPUT FILE ***********************")
    while True:
        line = ifi.readline()
        if not line:
            break
        print(line)
        words = line.split()
        if not words:
            continue
        if words[0][0] == "#":
            continue
        words[0] = words[0].lower()
        try:
            if words[0] == "coordinates":
                keywords['in_coords'] = words[1]
                continue
            if words[0] == "structure":
                keywords['topology'] = words[1]
                if len(words) > 2:
                    keywords['topology_type'] = words[2].lower()
                continue
            if words[0] == "groups_file":
                keywords['groups_path'] = words[1]
                continue
            if words[0] == "elements_file":
                keywords['elements_file_type'] = words[1]
                keywords['elements_file'] = words[2]
                continue
            if words[0] == "donor_index":
                keywords['donor_index'] = words[1]
                continue
            if words[0] == "output_coords":
                keywords['out_coords'] = words[1]
                continue
            if words[0] == "rlist":
                keywords['rlist'] = words[1]
                continue
            if words[0] == "rdh0":
                keywords["rdh0"] = words[1:]
                continue
            if words[0] == "indicator_verbose":
                keywords["indicator_verbose"] = True
                continue
            if words[0] == "write_n_steps":
                keywords["write_freq"] = int(words[1])
                continue
            if words[0] == "write_partitions":
                keywords["write_partitions"] = True
                if words[1]:
                    if words[1][-1] != '/':
                        words[1] += '/'
                    keywords["write_folder"] = words[1]
                    keywords["write_type"] = words[2]
                continue
            if words[0] == "write_prefix":
                keywords["write_prefix"] = words[1]
                continue
            if words[0] == "active_radius":
                keywords[words[0]] = float(words[1])
                continue
            if words[0] == "buffer_radius":
                keywords[words[0]] = float(words[1])
                continue
            if words[0] == "dimensions":
                keywords['dimensions'] = [float(words[1]), float(words[2]),
                                          float(words[3])]
                continue
            if words[0] == "pap_order":
                keywords['pap_order'] = int(words[1])
                continue
            if words[0] == "allow_hop":
                keywords['allow_hop'] = bool(int(words[1]))
                continue
            if words[0] == "dcd_pbc":
                keywords['dcd_pbc'] = bool(int(words[1]))
                continue
            if words[0] == "verbose":
                keywords['verbose'] = bool(int(words[1]))
                continue
            if words[0] == "ind_output_freq":
                keywords['ind_output_freq'] = int(words[1])
                continue
            if words[0] == "nowrap":
                keywords['wrap'] = False
                continue
            if words[0] == "ind_method":
                keywords['ind_method'] = int(words[1])
                continue
            if words[0] == "proton_types":
                keywords['proton_types'] = words[1:]
                continue
            if words[0] == "no_pre_topo_change":
                keywords['ratio_topology_change'] = False
                continue
            if words[0] == 'mcec':
                keywords['mcec'] = True
                keywords['mcec_w'] = words[:]
                continue
            if words[0] == 'rsw':
                keywords['rsw'] = float(words[1])
                continue
            if words[0] == 'dsw':
                keywords['dsw'] = float(words[1])
                continue
            if words[0] == 'mcec_g':
                keywords['mcec_g'] = words[1:]
                continue
            if words[0] == 'ap_write_probability':
                keywords['ap_write_probability'] = float(words[1])
                continue
            print("Could not find keyword: " + words[0])
            raise RuntimeError

        except RuntimeError:
            print("Error parsing keyword " + words[0])
            sys.exit()
    ifi.close()
    print("****************************************************")
    return keywords


def check_keywords(keywords, indi):
    """
    Check and print out the keywords and initial variables
    :param keywords: The parsed keywords
    :type keywords: dict
    :param indi: The indicator class used for the run
    :type indi: Indicator
    :return:
    """
    # Print I/O variables
    print("INPUT FILE VARIABLES")
    if "in_coords" in keywords:
        print("Reading coordinates from: " + keywords['in_coords'])
    else:
        print("Error, %s not found in keywords" % "'coordinates'")
        raise ValueError

    if keywords["topology"]:
        print("Reading topology from: " + keywords['topology'])
    else:
        print("Error, %s not found in keywords" % "'structure'")
        raise ValueError

    if keywords["out_coords"]:
        print("Writing coordinates to: " + keywords['out_coords'])
    else:
        print("Not writing output coordinates")

    if keywords["groups_path"]:
        print("Reading AP groups from: " + keywords['groups_path'])

    if keywords["elements_file"]:
        print("Reading atomic elements from: " + keywords['elements_file'])

    if keywords["write_partitions"]:
        if not keywords["groups_path"]:
            print("Error, you must specify groups_path with AP groups to write"
                  " AP partitons")
            sys.exit()
        print("Writing partitions to folder: %s" % keywords["write_folder"])
        print("Writing partitions every %d steps" % keywords["write_freq"])
    if keywords["wrap"] and not keywords["topology"]:
        sys.exit("Error, cannot wrap the system "
                 "because there is no topology specified")
    if keywords["ap_write_probability"]:
         if keywords["ap_write_probability"] < 0 or \
            keywords["ap_write_probability"] > 1:
             sys.exit("Error write probability must be between 0 and 1")
    # Print all of the indicator variables
    print("\nINDICATOR VARIABLES")
    print("rlist: {0:0.3f}".format(indi.rlist))
    print("rdh0 parameters:")
    for key in indi.rxh:
        print("    {0:8s}    {1:0.3f}".format(key, indi.rxh[key]))
    print("Initial donor: {0:d}".format(indi.donor))
    print("proton types:", keywords["proton_types"])

    # Print all of mCEC variables
    if keywords['mcec']:
        print("mCEC Variables")
        print("rsw: %.3f" % keywords['rsw'])
        print("dsw: %.3f" % keywords['dsw'])

    # Print general things
    print("SYSTEM VARIABLES")
    try:
        if 'dimensions' in keywords:
            print("Dimensions: {0:0.8f}   {1:0.8f}   {2:0.8f}".format(
                *keywords['dimensions']))
        elif keywords['wrap']:
            print("\n\nERROR: Please supply system dimensions as three floats"
                  " corresponding to x, y, z vectors")
            sys.exit()
        else:
            print("No periodic boundaries specified")
    except ValueError:
        print("Please supply system dimensions as three floats corresponding "
              "to x, y, z vectors")
        sys.exit()
    try:
        if keywords['write_partitions']:
            print("AP SETTINGS")
            print("Active radius: {0:0.8f}".format(keywords["active_radius"]))
            print("Buffer radius: {0:0.8f}".format(keywords["active_radius"]))
            print("Permute order: {0:1d}".format(keywords["pap_order"]))
    except ValueError:
        print("Error printing AP keywords")
        sys.exit()
    return


def set_indicator(keywords):
    """
    Initialize the indicator
    :param keywords: input keywords
    :type keywords: dict
    :return: indicator
    :rtype: Indicator
    """
    if keywords["ind_method"] == -1:
        indi = IndicatorNull()
    elif keywords["mcec"]:
        if keywords["ind_method"] in [4, 11]:
            indi = MCEC()
        else:
            print("Currently, you must only use mcec with indicator 4")
            raise NotImplementedError
    elif keywords["ind_method"] in [0, 1, 2, 8]:
        indi = Indicator()
    elif keywords["ind_method"] in [3, 4]:
        indi = Indicator4()
    elif keywords["ind_method"] in [6]:
        indi = Indicator6()
    elif keywords["ind_method"] in [7]:
        indi = Indicator7()
    elif keywords["ind_method"] in [9]:
        indi = Indicator9()
    elif keywords["ind_method"] in [11]:
        indi = Indicator11()
    else:
        print("Error, could not recognize indicator type")
        raise TypeError

    # Set the donor
    try:
        indi.donor = int(keywords["donor_index"])
    except ValueError:
        print("Error, please supply the initial donor index as an integer")

    # We can exit now if we are null class
    if keywords["ind_method"] == -1:
        return indi

    if keywords["ind_method"] in [2, 3]:
        print("Unfortunately the requested indicator type is not implemented")
        raise NotImplementedError

    if "rlist" in keywords:
        try:
            indi.rlist = float(keywords["rlist"])
        except ValueError:
            print("Error parsing rlist variable. It must be a float")
    if keywords["rdh0"]:
        vals = keywords["rdh0"]
        if len(vals) % 2 != 0:
            print("Error parsing rdh0 parameters, there should be an even "
                  "number of them")
            raise ValueError
        for i in range(len(vals) // 2):
            try:
                indi.rxh[vals[2*i]] = float(vals[2*i + 1])
            except ValueError:
                print("Error parsing rdh0 variables: ", vals[2*i:2*i+1])
    else:
        print("Error, did you specify the rdh0 keyword?")
        raise NameError

    try:
        indi.print_all = keywords["indicator_verbose"]
    except ValueError:
        print("Error setting indicator.printall flag")
    indi.set_output_freq(keywords["ind_output_freq"], keywords['write_prefix'])

    if keywords['mcec']:
        initialize_mcec(keywords, indi)
    return indi


def initialize_mcec(keywords, indi):
    """
    Initialize the mCEC variables for the mCEC indicator. Note that currently
    you must have initialized Indicator4 variables previously to initializing
    the mCEC.
    :param keywords: dict user input
    :param indi:
    :return:
    """
    # Parse the acceptor type keywords. Make sure they are also acceptors for
    # Indicator 4
    vals = keywords["mcec_w"][:]
    if len(vals) == 1:
        print("Error, no acceptor types found after keyword 'mcec'")
        raise NameError
    vals = vals[1:]
    if len(vals) % 2 != 0:
        print("Error, each acceptor type for keyword 'mcec' must be followed by"
              " integer")
        print("The integer is the reference state for the least protonated"
              " state of the atom")
        raise NameError
    for i in range(len(vals) // 2):
        acc = vals[2*i]
        w = vals[2*i+1]
        if acc not in indi.rxh.keys():
            print()
            print("Error, mCEC type found but does not correspond with "
                  " the acceptors in the rdh0 list")
            raise
        try:
            indi.m_acc_weight[acc] = float(w)
        except TypeError:
            print("Error, weight is not float for acceptor %s" % acc)
            print("Error keyword mcec")
            raise

    for key in list(indi.rxh.keys()):
        if key not in indi.m_acc_weight.keys():
            print("Error, acceptor %s is not in mcec list but is in rxh list"
                  % key)
            raise KeyError
    indi.rsw = keywords['rsw']
    indi.dsw = keywords['dsw']
    vals = keywords['mcec_g']
    if vals:
        for val in vals:
            words = val.split(',')
            if len(words) <= 2:
                print("For keyword mcec_g")
                print("Error parsing mcec_g keyword, there are not enough"
                      " integers for the group")
                raise TypeError
            group_ids = []
            for w in words[:-1]:
                if not w.isdigit():
                    print("For keyword mcec_g")
                    print("Error parsing integer for group id")
                    raise TypeError
                group_ids.append(int(w))
            if not words[-1].isdigit():
                print("For keyword mcec_g")
                print("Error, group weight must be an integer corresponding to"
                      " the reference state")
                raise TypeError
            indi.correction_groups.append(group_ids)
            indi.correction_weights.append(int(words[-1])/float(len(group_ids)))
    return


def cleanup():
    """
    This subroutine deletes any temporary files which have been created during
    program execution.

    Parameters
    ----------

    Returns
    -------

    """
    if os.path.isfile('tmp.mol2'):
        os.remove('tmp.mol2')


def main():
    """
    This is the main hook. Here we parse the input, initialize data structures,
    and call the main loop over the trajectory.

    Parameters
    ----------

    Returns
    -------

    """
    debug = False
    arg_vals = None
    if debug:
        arg_vals = ['-i', 'ind_tools.inp']
    command_line_args = get_args(arg_vals)

    ifpath = command_line_args.input
    keywords = read_input_file(ifpath)

    indi = set_indicator(keywords)
    check_keywords(keywords, indi)
    core_loop(keywords, indi)
    cleanup()


if __name__ == "__main__":
    main()
