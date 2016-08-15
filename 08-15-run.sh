#!/bin/bash

echo ${1?Need to specify an input file!} 1>/dev/null
SRCDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $SRCDIR
if [ ! -e out ]; then
    mkdir out
fi

xyz_in=$SRCDIR/out/bulk08.xyz
lmp_in=$SRCDIR/out/bulk08.data
lmpconfig=$SRCDIR/out/configbulk08.lmp

python build_lat.py -scale 2.66 -nx 6 -ny 6 -nz 8 -noiz 0.001 -ndefect 1 -fmt xyz > $xyz_in
python build_lat.py -scale 2.66 -nx 6 -ny 6 -nz 8 -noiz 0.001 -ndefect 1 -fmt lmp > $lmp_in

# echo << LAMMPSIN > $lmpconfig
# pair_style lj/cut 2.5
# LAMMPSIN

if [ ! $(uname -s) == "Darwin" ]; then
    vmd $xyz_in
    sed -i 's/cdse.*/cdse.data/g' $1
    lammps     -in $1 
    rm log.lammps
else
    lammps -in nvt.inp
fi

