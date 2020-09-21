#!/bin/bash
#/*
# * Copyright (c) 2020, Mariano Rodriguez <rdguez.mariano@gmail.com>
# * All rights reserved.
# *
# * This program is free software: you can use, modify and/or
# * redistribute it under the terms of the GNU General Public
# * License as published by the Free Software Foundation, either
# * version 3 of the License, or (at your option) any later
# * version. You should have received a copy of this license along
# * this program. If not, see <http://www.gnu.org/licenses/>.
# */

set -e

virtualenv=$1
demoextrasfolder=$2
binfolder=$3
input0="input_0.png"
input1="input_1.png"
gfilter=$4
covering=$5
type=$6
detector=$7
descriptor=$8


if [ -d $virtualenv ]; then
  source $virtualenv/bin/activate
fi
# echo "$virtualenv"
# echo "$binfolder"
# pwd

cp $binfolder/imas_bin imas_bin
# ls -al $binfolder
# ls -al

workfolder="/tmp/$$/"
mkdir -p $workfolder

adimas_caller.py --im1 $input0 --im2 $input1 --gfilter $gfilter --covering $covering --type $type --detector $detector --descriptor $descriptor --workdir $workfolder --bindir $binfolder

rm imas_bin

# ls -al
