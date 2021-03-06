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
    3. Three classes are made: down, flat and up based on market motion by 'thresh_motion' cents after 'thresh_time' seconds
    '''
    ### collect features
    num_features = 17
    scaler = [1.0, 1.0e5, 1.0e3, 1.0e5, 1.0e3, 1.0e5, 1.0e3, 1.0e5, 1.0e3, 
                   1.0e5, 1.0e3, 1.0e5, 1.0e3, 1.0e5, 1.0e3, 1.0e5, 1.0e3] 
    thresh_motion = 50
    thresh_time   = 300

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
            ## add to list 'day'
            day.append(line)

    ### create labels
    ## based on 300 second future
    ## one-hot vector. [down, flat, up].
    ind      = 0
    ind_fut  = 0
    tot_inds = len(day)
    labels   = []
    while ind < tot_inds:
        midpoint = 0.5*(day[ind][1] + day[ind][3])
        while day[ind_fut][0] < day[ind][0] + thresh_time:
            if ind_fut < tot_inds-1:
                ind_fut += 1
            else:
                break
        midpoint_fut = 0.5*(day[ind_fut][1] + day[ind_fut][3])
        if midpoint + thresh_motion <= midpoint_fut: # midpoint motion up
            labels.append([0, 0, 1])
        elif midpoint - thresh_motion >= midpoint_fut: # midpoint motion down
            labels.append([1, 0, 0])
        else:
            labels.append([0, 1, 0])
        ind += 1
            
    ### scale day
    day = [[d[i]/scaler[i] for i in range(num_features)] for d in day]

    ### map to numpy arrays
    day = np.asarray(day)
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

