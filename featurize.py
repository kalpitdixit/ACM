import glob
import os
import numpy as np

def convert_to_time(tstamp):
    tstamp = tstamp.split(':')
    return float(tstamp[0])*3600 + float(tstamp[1])*60 + float(tstamp[2])
    

def featurize_file(fname):           
    # collect features
    day = []
    first_time = None
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split(',')[:17]
            if first_time is None:
                first_time = convert_to_time(line[0])
            line[0] = convert_to_time(line[0])
            if first_time <= line[0]:
                line[0] = line[0] - first_time # time in seconds since start
            else:
                line[0] = line[0] + 84600 - first_time # time in seconds since start
            line = map(float, line)
            day.append(line)
    day = np.asarray(day)

    ### create labels
    ## based on 300 second future
    ind      = 0
    ind_60   = 0
    tot_inds = day.shape[0]
    labels = []
    while ind < tot_inds:
        midpoint = 0.5*(day[ind][1] + day[ind][3])
        while day[ind_60][0] < day[ind][0] + 300:
            if ind_60 < tot_inds-1:
                ind_60 += 1
            else:
                break
        midpoint_60 = 0.5*(day[ind_60][1] + day[ind_60][3])
        if midpoint + 50 <= midpoint_60: # midpoint motion
            labels.append(1)
        elif midpoint - 50 >= midpoint_60: # midpoint motion
            labels.append(-1)
        else:
            labels.append(0)
        ind += 1

    return day, labels


if __name__=="__main__":
    
    dest_data_dir = '/afs/.ir/users/k/a/kalpit/ACM_data'
    fnames = glob.glob('/afs/.ir/users/k/a/kalpit/ACM_raw_data/*csv')
    count = 0
    
    for fname in fnames:
        feats, labels = featurize_file(fname)    
        feats  = feats[:10]
        labels = labels[:10]
        np.save(os.path.join(dest_data_dir, 'feats{}.npy'.format(count)), feats)
        np.save(os.path.join(dest_data_dir, 'labels{}.npy'.format(count)), labels)
        count += 1
    
    # num_files
    with open(os.path.join(dest_data_dir, 'num_files'), 'w') as f:
        f.write(str(count))

