

import sys
import h5py
import numpy as np
import pyBigWig as pbw
import re

if __name__ == '__main__':
	
	dataname = re.split('.bed',sys.argv[1])[0]
	
	#BED file
	fbed = open(sys.argv[1],'r')
	
	chrmlentot = np.array([249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566])
	res = 200. #Le point est important pour la dicvision qui suit!!!
	chrmlentotbin = np.cumsum(np.ceil(chrmlentot/res))
	chrmlentotbin = np.insert(chrmlentotbin, 0, 0)
	S = int(np.sum(np.ceil(chrmlentot/res)))
	print S
	vect = np.zeros((S,1))
	
	for line in fbed:
			if re.search(r'chr(\d+)', line):
			    #print (line.split('\n')[0]).split('\t')
			    values = (line.split('\n')[0]).split('\t')
		
			    chrm = int((values[0]).split('chr')[1])
			    pos1 = int(chrmlentotbin[chrm-1] + np.floor(int(values[1])/res))
			    pos2 = int(chrmlentotbin[chrm-1] + np.floor(int(values[2])/res))
                            if pos1 != pos2:
                                    vect[range(pos1,pos2)] = vect[range(pos1,pos2)] + float(values[3])
			    else:
                                    vect[range(pos1,pos1+1)] = vect[range(pos1,pos1+1)] + float(values[3])

	fbed.close()

	fh5 = h5py.File(dataname + '.hdf5', "w")
	fh5['data'] = vect
	fh5.close()
