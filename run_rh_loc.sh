#!/bin/sh

filename="./result/fig3_3/bs_tfq_loc.csv"

for wx in -5.0
  do
   python3 rh_file.py ${wx} ${filename}
  for velo in -2.0 -1.9 -1.8 -1.7 -1.6 -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
   do
    python3 RH_exp01_loc.py ${wx} ${velo} ${filename}
   done
 done
