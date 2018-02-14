
import sys
import os.path
import numpy as np
import lmdb
import caffe
import random
import cv2

caffe_root = '/home/enroutelab/Amy/caffe' 
directory = '/media/enroutelab/sdd/data/image2gps/BigSampleStreetView/'
dataset = 'data.txt'

poses = []
images = []

sys.path.insert(0, caffe_root + 'python')

with open(directory+dataset) as f:
    for _ in xrange(16):
        next(f)
    for line in f:
        linesplit = line.split()
        fname = directory + "satellite/" + linesplit[3]
        if(os.path.isfile(fname) & (float(linesplit[4])==1)):
            lat = float(linesplit[0])
            lon = float(linesplit[1])
            poses.append((lat, lon))
            images.append(fname)

print("There are " + str(len(poses)) + " images on grid.")

r = list(range(len(images)))
random.shuffle(r)

env = lmdb.open('dataset_lmdb', map_size=int(1e12))

count = 0
for i in r:
    if (count + 1) % 100 == 0:
        print 'Saving images: ', count + 1
    X = cv2.imread(images[i])
    X = cv2.resize(X, (455,256))   
    X = np.transpose(X,(2,0,1))
    im_dat = caffe.io.array_to_datum(np.array(X).astype(np.uint8))
    im_dat.float_data.extend(poses[i])
    str_id = '{:0>10d}'.format(count)
    with env.begin(write=True) as txn:
        txn.put(str_id, im_dat.SerializeToString())
    count = count+1

env.close()

