import glob 
import os
import numpy as np

class Loader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def convert_to_time(self, tstamp):
        tstamp = tstamp.split(':')
        return float(tstamp[0])*3600 + float(tstamp[1])*60 + float(tstamp[2])

    def iterate(self):
        with open(os.path.join(self.data_dir, 'num_files')) as f:
            num_files = int(f.readline())

        for i in xrange(num_files):
            feats  = np.load(os.path.join(self.data_dir, 'feats{}.npy'.format(i)))
            labels = np.load(os.path.join(self.data_dir, 'labels{}.npy'.format(i)))
            feats  = np.expand_dims(feats, 0)
            yield feats, labels
