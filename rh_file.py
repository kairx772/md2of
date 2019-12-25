import sys
# with open('./result/rh_tfq.csv', 'a') as f:
#     f.write('velo,wx,-6.0,-5.5,-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0\n')

wx = float(sys.argv[1])
filename = sys.argv[2]

with open(filename, 'a') as f:
    f.write('\n,'+str(wx)+',')
