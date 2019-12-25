#!/bin/sh

filename="./result/fig3_2/rh_tcons_2.csv"

for velo in 3.0 3.5 4.0 
  do
   python3 rh_file.py ${velo} ${filename}
  for tcons in 0.05 0.15 0.25 0.35 0.45 
   do
    python3 RH_exp01_tcons.py ${velo} ${tcons} ${filename}
   done
 done


# -6.0 -5.5 -5.0 -4.5 -4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 2.5
