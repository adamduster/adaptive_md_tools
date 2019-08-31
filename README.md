adaptive_md_tools
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/adaptive_md_tools.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/adaptive_md_tools)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/REPLACE_WITH_OWNER_ACCOUNT/adaptive_md_tools/branch/master)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/adaptive_md_tools/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/adaptive_md_tools/branch/master)

This package provides tools for analyizing adaptive partitioning simulations, especially those with protons.

## Abstract
Adaptive MD Tools is a package that aims to help with getting partitions from completed MD simulations.
The user supplies a topology file and trajectory file, and then selects the active zone center.
The active zone center can be an individual atom, or one of several proton tracking algorithms including our indicator and the modified center of excess charge (mCEC).
The program can then create a new trajectory where the active center is placed in the center of the box and the other atoms are wrapped around it based on supplied unit cell dimensions for an NVT simulation.
The user can calculate the partitions from the permuted AP algorithm for a given permute order and buffer zone radius, and output them as .xyz  files to a folder.

## Installation
For complete installation instructions, please read the program documentation at ./docs/manual.pdf

### Copyright

Copyright (c) 2019, Adam Duster


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
