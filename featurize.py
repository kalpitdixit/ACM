import glob
import os
import numpy as np

DTYPE = np.float32

def convert_to_time(tstamp):
    tstamp = tstamp.split(':')
    return float(tstamp[0])*3600 + float(tstamp[1])*60 + float(tstamp[2])
    

def featurize_file(fname):           

    '''
    Many special things are done since we are processing an Order Book
    1. Time is modified to be delta since last entry, since that matters most
    2. All dimensions are scaled by variable 'scaler' which is done to avoid large numbers in the features without losing scale information.
    '''
    ### collect features
    num_features = 17
    scaler = [1.0, 1.0e5, 1.0e3, 1.0e5, 1.0e3, 1.0e5, 1.0e3, 1.0e5, 1.0e3, 
                   1.0e5, 1.0e3, 1.0e5, 1.0e3, 1.0e5, 1.0e3, 1.0e5, 1.0e3] 

    assert len(scaler)==num_features, 'len(scaler)!=num_features'
    day = []
    first_time = None
    prev_time = 0.0
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split(',')[:num_features]

            ## process time dimension
            # get elapsed time since start of trading day
            if first_time is None:
                first_time = convert_to_time(line[0])
            line[0] = convert_to_time(line[0])
            if first_time <= line[0]:
                line[0] = line[0] - first_time # time in seconds since start
            else:
                line[0] = line[0] + 84600 - first_time # time in seconds since start

            # get time since last entry
            line[0] = line[0] - prev_time
            prev_time = prev_time + line[0]
            
            ## make everything a float and scale it    
            line = map(float, line)
            line = [line[i]/scaler[i] for i in range(num_features)]
            ## add to list 'day'
            day.append(line)
    day = np.asarray(day)

    ### create labels
    ## based on 300 second future
    ## one-hot vector. [down, flat, up].
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
            labels.append([1, 0, 0])
        elif midpoint - 50 >= midpoint_60: # midpoint motion
            labels.append([0, 1, 0])
        else:
            labels.append([0, 0, 1])
        ind += 1
    labels = np.asarray(labels)
    
    return day, labels


if __name__=="__main__":
    
    dest_data_dir = '/afs/.ir/users/k/a/kalpit/ACM_data'
    fnames = glob.glob('/afs/.ir/users/k/a/kalpit/ACM_raw_data/*csv')
    count = 0
    
    for fname in fnames:
        feats, labels = featurize_file(fname)    
        feats  = feats.astype(DTYPE)
        labels = labels.astype(DTYPE)

        ## save features files
        np.save(os.path.join(dest_data_dir, 'feats{}.npy'.format(count)), feats)
        np.save(os.path.join(dest_data_dir, 'labels{}.npy'.format(count)), labels)
        count += 1
    
    # num_files
    with open(os.path.join(dest_data_dir, 'num_files'), 'w') as f:
        f.write(str(count))

