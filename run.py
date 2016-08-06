import theano
import theano.tensor as T
import keras
from keras import backend as K
import lasagne

from loader import Loader

from keras.layers import Input, SimpleRNN, LSTM, Dense, TimeDistributed, BatchNormalization, Activation, Reshape, Flatten
from keras.models import Model

from keras.backend import categorical_crossentropy

import numpy as np
import random

class CFG:
    epochs = 20
    input_dim = 17
    recur_layers = 1
    nodes = 5
    init = 'glorot_uniform'
    output_dim = 3
    lr = 1e-2
    use_LSTM = True

def simple_LSTM_model(cfg=CFG()):
    ob_input = Input(shape=(None, cfg.input_dim), name='ob_input')

    prev_layer = ob_input
    for r in range(cfg.recur_layers):
        if cfg.use_LSTM:
            prev_layer = LSTM(cfg.nodes, name='lstm_{}'.format(r+1), init=cfg.init, return_sequences=True, consume_less='gpu')(prev_layer)
        else:
            prev_layer = SimpleRNN(cfg.nodes, name='lstm_{}'.format(r+1), init=cfg.init, return_sequences=True)(prev_layer)
        last_rnn = prev_layer
        prev_layer = BatchNormalization(name='lstm_bn_{}'.format(r+1), mode=0, axis=2)(prev_layer)
        prev_layer = Activation('relu')(prev_layer)

    network_outputs = TimeDistributed(Dense(cfg.output_dim, name='network_outputs', init=cfg.init, activation='linear'))(prev_layer)
    raw_softmax_outputs = Activation('softmax')(network_outputs)
    
    model = Model(input=ob_input, output=[raw_softmax_outputs])

    return model    


def build_train_fn(model):
    # cost
    lr = T.scalar()
    labels = K.placeholder(ndim=2, dtype='int32')
    ob_input = model.inputs[0]
    raw_softmax_outputs = model.outputs[0]

    softmax_outputs = raw_softmax_outputs.dimshuffle((2,0,1))
    softmax_outputs = softmax_outputs.reshape((softmax_outputs.shape[0], softmax_outputs.shape[1]*softmax_outputs.shape[2]))
    softmax_outputs = softmax_outputs.dimshuffle((1,0))

    cost = categorical_crossentropy(softmax_outputs, labels).mean()

    # gradients
    trainable_vars = model.trainable_weights
    grads = K.gradients(cost, trainable_vars)
    grads = lasagne.updates.total_norm_constraint(grads, 100)
    updates = lasagne.updates.nesterov_momentum(grads, trainable_vars, lr, 0.99)

    for key, val in model.updates:                              
        updates[key] = val

    # train_fn
    train_fn = K.function([ob_input, labels, K.learning_phase(), lr],
                          [softmax_outputs, cost],
                          updates=updates)

    return train_fn


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
        accuracy.append(round(100.0*correct[-1]/total[-1],3)) 

    return correct, total, accuracy


def train(train_fn, dataset, cfg=CFG()):
    for e in range(cfg.epochs):
        for i, (feats, labels) in enumerate(dataset.iterate()):

            softmax_outputs, cost = train_fn([feats, labels, True, cfg.lr])
            correct, total, accuracy = get_accuracy(softmax_outputs, labels)

            print 'Epochs: {}  Cost: {:.4f}  correct:{}  total:{}  accuracy:{}'.format(e, float(cost), correct, total, accuracy)

    return


if __name__=="__main__":
    np.random.seed(1)
    random.seed(1)

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
    train(train_fn, dataset, cfg)










