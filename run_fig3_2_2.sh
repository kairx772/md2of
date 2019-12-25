#!/bin/sh

for wx in -5.5
  do
  for velo in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

   do
    python3 RH_fig3_2_2.py ${wx} ${velo}
   done
 done
