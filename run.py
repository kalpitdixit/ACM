import tensorflow as tf
import keras
#from keras import backend as K
from loader import Loader

#from keras.layers import Input, LSTM, Dense, TimeDistributed, BatchNormalization, Activation
#from keras.models import Model

class CFG:
    epochs = 1    
    input_dim = 17
    recur_layers = 1
    nodes = 10
    init = 'glorot_uniform'
    output_dim = 3
    lr = 1e-4

def simple_LSTM_model(cfg=CFG()):
    ob_input = Input(shape=(cfg.input_dim), name='ob_input')

    prev_layer = ob_input
    for r in range(cfg.recur_layers):
        prev_layer = LSTM(cfg.nodes, name='lstm_{}'.format(r+1), init=cfg.init, return_sequences=True, consume_less='gpu')(prev_layer)
        prev_layer = Activation('relu')(BatchNormalization(name='lstm_bn_{}'.format(r+1))(prev_layer))

    network_outputs = TimeDistributed(Dense(cfg.output_dim, name='network_outputs', init=cfg.init, activation='linear'))(prev_layer)
    softmax_outputs = Activation('softmax')(network_outputs)
    
    model = Model(input=ob_input, output=softmax_outputs)

    return model    


def train_fn(model):
    print '##### Training Neural Network #####'
    # cost
    lr = tf.scalar()
    labels = K.placeholder(ndim=1, dtype='int32')
    ob_input = model.inputs[0]
    softmax_outputs = model.outputs[0]
    cost = categorical_crossentropy(labels, softmax_outputs)

    # gradients
    trainable_vars = model.trainable_weights
    grads = K.gradients(ce_cost, trainable_vars)
    grads = lasagne.updates.total_norm_constraint(grads, 100)
    updates = lasagne.updates.nesterov_momentum(grads, trainable_vars, lr,
                                                0.99)
    for key, val in model.updates:                              
        updates[key] = val

    # train_fn
    train_fn = K.function([ob_input, labels,
                           K.learning_phase(), lr],
                          [softmax_outputs, cost],
                          updates=updates)

    return train_fn


def train(train_fn, dataset, cfg=CFG()):
    for e in range(cfg.epochs):
        for i, (feats, labels) in enumerate(dataset.iterate()):
            _, cost = train_fn([feats, labels, True, cfg.lr])
            print 'Epochs: {}  Cost: {:.4f}'.format(e, cost)
    return


if __name__=="__main__":
    print 'asf' 
    cfg = CFG()
    print 'asf' 
    train_data_dir = '/afs/.ir/users/k/a/kalpit/ACM_data/'

    print 'asf' 
    dataset = Loader(train_data_dir)
    print 'asf'

    # Training Neural Network
    train(train_fn, dataset, cfg)
