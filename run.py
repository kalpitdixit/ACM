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


class CFG:
    epochs = 1000
    input_dim = 17
    recur_layers = 3
    nodes = 10
    init = 'glorot_uniform'
    output_dim = 3
    lr = 1e-3
    use_LSTM = True


def simple_LSTM_model(cfg=CFG()):
    ob_input = Input(shape=(None, cfg.input_dim), name='ob_input')

    prev_layer = ob_input
    for r in range(cfg.recur_layers):
        if cfg.use_LSTM:
            prev_layer = LSTM(cfg.nodes, name='lstm_{}'.format(r+1), init=cfg.init, return_sequences=True, consume_less='gpu')(prev_layer)
        else:
            prev_layer = SimpleRNN(cfg.nodes, name='lstm_{}'.format(r+1), init=cfg.init, return_sequences=True)(prev_layer)
        prev_layer = BatchNormalization(name='lstm_bn_{}'.format(r+1), mode=0, axis=2)(prev_layer)
        prev_layer = Activation('relu')(prev_layer)

    network_outputs = TimeDistributed(Dense(cfg.output_dim, name='network_outputs', init=cfg.init, activation='linear'))(prev_layer)
    raw_softmax_outputs = Activation('softmax')(network_outputs)
    
    model = Model(input=ob_input, output=raw_softmax_outputs)

    return model    


def build_train_fn(model):
    ### cost
    lr = T.scalar()
    labels  = K.placeholder(ndim=2, dtype='int32')
    weights = K.placeholder(ndim=1, dtype='float32')

    ob_input = model.inputs[0]
    raw_softmax_outputs = model.outputs[0]

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
                          [softmax_outputs, cost],
                          updates=updates)

    return train_fn


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
        

def train(train_fn, dataset, run_dir, cfg=CFG()):
    train_costs = []
    for e in range(cfg.epochs):
        for i, (feats, labels, weights) in enumerate(dataset.iterate()):

            softmax_outputs, cost = train_fn([feats, labels, weights, True, cfg.lr])
            correct, total, accuracy = get_accuracy(softmax_outputs, labels)
            conf_mat = get_confusion_matrix(softmax_outputs, labels)

            print 'Epochs: {}  Cost: {:.4f}  correct:{}  total:{}  accuracy:{}'.format(e, float(cost), correct, total, accuracy)
            print conf_mat
            print ''

            # save metrics
            train_costs.append(cost)
            np.save(os.path.join(run_dir, 'train_costs.npy'), train_costs)

    return


if __name__=="__main__":
    np.random.seed(1)
    random.seed(1)

    # run_id
    run_id = get_run_id()
    run_dir = 'run_{}'.format(run_id)
    os.system('mkdir '+run_dir)

    # params
    cfg = CFG()
    train_data_dir = '/afs/.ir/users/k/a/kalpit/ACM_data/'
    dataset  = Loader(train_data_dir)

    # Model
    print '##### Building Model #####'
    model = simple_LSTM_model(cfg)

    # Train Function
    print '##### Building Train Function #####'
    train_fn = build_train_fn(model) 

    # Training Neural Network
    print '##### Training Neural Network #####'
    train(train_fn, dataset, run_dir, cfg)
