import theano
import theano.tensor as T
import keras
from keras import backend as K
import lasagne

from loader import Loader

from keras.layers import Input, SimpleRNN, LSTM, Dense, TimeDistributed, BatchNormalization, Activation, Reshape, Flatten
from keras.models import Model
from keras.engine.training import weighted_objective

from keras.backend import categorical_crossentropy

import numpy as np
import random
import os
import pickle


class CFG:
    epochs = 1000
    input_dim = 17
    recur_layers = 1
    nodes = 10
    init = 'glorot_uniform'
    output_dim = 3
    lr = 1e-2
    use_LSTM = True
    
    def save(self, save_dir):
        config_filename = os.path.join(save_dir, 'config.pkl')
        with open(config_filename, 'w') as f:
            pickle.dump(self.__dict__, f, 2)

    def load(self, save_dir):
        config_filename = os.path.join(save_dir, 'config.pkl')
        if os.path.exists(config_filename):
            with open(config_filename) as f:
                obj = pickle.load(f)
                self.__dict__.update(obj)


def simple_LSTM_model(cfg=CFG()):
    ob_input = Input(shape=(None, cfg.input_dim), name='ob_input')

    prev_layer = ob_input
    for r in range(cfg.recur_layers):
        if cfg.use_LSTM:
            prev_layer = LSTM(cfg.nodes, name='lstm_{}'.format(r+1), init=cfg.init, return_sequences=True, consume_less='gpu')(prev_layer)
        else:
            prev_layer = SimpleRNN(cfg.nodes, name='lstm_{}'.format(r+1), init=cfg.init, return_sequences=True)(prev_layer)
        pre_bn = prev_layer
        prev_layer = BatchNormalization(name='lstm_bn_{}'.format(r+1), mode=0, axis=2)(prev_layer)
        post_bn = prev_layer
        prev_layer = Activation('relu')(prev_layer)

    network_outputs = TimeDistributed(Dense(cfg.output_dim, name='network_outputs', init=cfg.init, activation='linear'))(prev_layer)
    raw_softmax_outputs = Activation('softmax')(network_outputs)
    
    model = Model(input=ob_input, output=[raw_softmax_outputs, post_bn])
    model.summary()

    return model    


def build_train_fn(model):
    ### cost
    lr = T.scalar()
    labels  = K.placeholder(ndim=2, dtype='int32')
    weights = K.placeholder(ndim=1, dtype='float32')

    ob_input = model.inputs[0]
    raw_softmax_outputs = model.outputs[0]
    prev_layer = model.outputs[1]

    softmax_outputs = raw_softmax_outputs.dimshuffle((2,0,1))
    softmax_outputs = softmax_outputs.reshape((softmax_outputs.shape[0], softmax_outputs.shape[1]*softmax_outputs.shape[2]))
    softmax_outputs = softmax_outputs.dimshuffle((1,0))

    #cost = categorical_crossentropy(softmax_outputs, labels).mean()
    wcc = weighted_objective(categorical_crossentropy)
    cost = wcc(softmax_outputs, labels, weights).mean()

    ### gradients
    trainable_vars = model.trainable_weights
    grads = K.gradients(cost, trainable_vars)
    grads = lasagne.updates.total_norm_constraint(grads, 100)
    updates = lasagne.updates.nesterov_momentum(grads, trainable_vars, lr, 0.99)

    for key, val in model.updates:                              
        updates[key] = val

    ### train_fn
    train_fn = K.function([ob_input, labels, weights, K.learning_phase(), lr],
                          [softmax_outputs, cost, prev_layer],
                          updates=updates)

    return train_fn


def get_job_id():
    if os.path.isfile('jobs/job_id'):
        with open('jobs/job_id', 'r') as f:  
            job_id = 1 + int(f.readline())
    else:
        job_id = 1
    with open('jobs/job_id', 'w') as f:  
        f.write(str(job_id))
    return job_id


def get_accuracy(softmax_outputs, labels):
    num_classes = labels.shape[1]

    pred_labels = np.argmax(softmax_outputs, axis=1)
    labels = np.argmax(labels, axis=1)

    correct  = []
    total    = []
    accuracy = []
    net_acc  = []
    for c in range(num_classes):
        ind = np.where(labels==c)[0]
        correct.append(np.sum(pred_labels[ind]==labels[ind]))
        total.append(labels[ind].shape[0])
        if total[-1]>0:
            accuracy.append(round(100.0*correct[-1]/total[-1],3)) 
        else:
            accuracy.append(round(-1.00,2))
    net_acc = round(100.0*sum(correct)/sum(total),3)
    return correct, total, accuracy, net_acc


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
        

def train(train_fn, dataset, job_dir, cfg=CFG()):

    # save config
    cfg.save(job_dir)
    
    # accuracies file
    with open(os.path.join(job_dir, 'accs.csv'), 'w') as f:
        f.write('[corrects, totals, accuracies]-net_acc-cost : each having as many entries as label classes\n')

    # train
    train_costs = []
    for e in range(cfg.epochs):
        for i, (feats, labels, weights) in enumerate(dataset.iterate()):

            softmax_outputs, cost, prev_layer = train_fn([feats, labels, weights, True, cfg.lr])
            correct, total, accuracy, net_acc = get_accuracy(softmax_outputs, labels)
            conf_mat = get_confusion_matrix(softmax_outputs, labels)

            print 'Epoch: {}/{}  job_dir: {}  Cost: {:.4f}  correct:{}  total:{}  accuracy:{}  net_acc:{}'.format(e, cfg.epochs, 
                  job_dir, float(cost), correct, total, accuracy, net_acc)
            print 'Num NaN in prev_layer: {}'.format(np.sum(np.isnan(prev_layer)))
            print prev_layer.shape
            print conf_mat
            print ''

            # save costs
            train_costs.append(cost)
            np.save(os.path.join(job_dir, 'train_costs.npy'), train_costs)
            # save accuracies
            with open(os.path.join(job_dir, 'accs.csv'), 'a') as f:
                f.write(','.join([str(_) for _ in correct])  + ',')
                f.write(','.join([str(_) for _ in total])    + ',')
                f.write(','.join([str(_) for _ in accuracy]) + ',')
                f.write(','.join([str(net_acc)]) + ',')
                f.write(','.join([str(cost)]))
                f.write('\n')
            
    return


if __name__=="__main__":
    np.random.seed(1)
    random.seed(1)

    # job_id and job_dir
    if not os.path.exists('jobs'):
        os.system('mkdir jobs')
    job_id = get_job_id()
    job_dir = 'jobs/job_{}'.format(job_id)
    os.system('mkdir '+job_dir)

    # params
    cfg = CFG()
    train_data_dir = '/afs/.ir/users/k/a/kalpit/ACM/ACM_data/'
    dataset  = Loader(train_data_dir)

    # Model
    print '##### Building Model #####'
    model = simple_LSTM_model(cfg)

    # Train Function
    print '##### Building Train Function #####'
    train_fn = build_train_fn(model) 

    # Training Neural Network
    print '##### Training Neural Network #####'
    train(train_fn, dataset, job_dir, cfg)
