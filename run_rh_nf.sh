#!/bin/sh

for wx in -8.0 -7.5 -6.0 -6.5 -6.0 -5.5 -5.0 -4.5 -4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0
  do
   python3 nf_file.py ${wx}
  for velo in -6.0 -5.5 -5.0 -4.5 -4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 
   do
    python3 RH_exp01_nf.py ${wx} ${velo}
   done
 done
