import numpy as np
import os


def get_run_id():
    if os.path.isfile('run_id'):
        with open('run_id', 'r') as f:  
            run_id = 1 + int(f.readline())
    else:
        run_id = 1
    with open('run_id', 'w') as f:  
        f.write(str(run_id))
    return run_id


def get_accuracy(softmax_outputs, labels):
    num_classes = labels.shape[1]

    pred_labels = np.argmax(softmax_outputs, axis=1)
    labels = np.argmax(labels, axis=1)

    correct  = []
    total    = []
    accuracy = []

    for c in range(num_classes):
        ind = np.where(labels==c)[0]
        correct.append(np.sum(pred_labels[ind]==labels[ind]))
        total.append(labels[ind].shape[0])
        if total[-1]>0:
            accuracy.append(round(100.0*correct[-1]/total[-1],3)) 
        else:
            accuracy.append(round(-1.00,2))

    return correct, total, accuracy


def get_confusion_matrix(softmax_outputs, labels):
    num_classes = labels.shape[1]
    conf_mat = np.zeros((num_classes, num_classes)) # correct x predicted
    pred_labels = np.argmax(softmax_outputs, axis=1)
    labels = np.argmax(labels, axis=1)
    
    for i in range(num_classes):
        for j in range(num_classes):
            ind = labels==i
            ind = pred_labels[ind]==j
            conf_mat[i][j] = np.sum(ind)
    
    return conf_mat.astype(np.int32)
