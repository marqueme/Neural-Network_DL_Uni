#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
from ast import literal_eval
import matplotlib.pyplot as plt

# range of sentence lengths
max_length = 50
min_length = 2
num_feats = 300

def load_text():
    # returns a dictionary of the structure:
    # text_dict = {part1: [utt1, utt2, utt3], part2: [...], ...}
    path = '/mount/studenten/deeplearning/2016/projects/SpeechTime/data/data_text/'
    text_dict = {}
    for part in ['train', 'dev', 'test']:
        text_dict[part] = {}
        with open(path + part) as utts:
            # read the utterances one by one
            # utt = [label, [vec_word1], [vec_word2], ...]
            for line in utts:
                ID = literal_eval(line)[0]
                utt = literal_eval(line)[1:]
                text_dict[part][ID] = utt
    return text_dict


def vectorize_data(data, num_utts):
    # input: dictionary of matrices
    # output: a 4D tensor (num_utts, 1, max_length, num_feats)
    vecData = {}
    for part in data:
        num = num_utts[part]
        X_part = "X_" + part
        y_part = "y_" + part
        x = [] #list of 50x300 matrices representing each utterance
        y = [] #list of labels, one for each utterance
        for utt in data[part]:
            y.append(data[part][utt][0])
            matrix = np.array(data[part][utt][1])
            # Padding
            rows = matrix.shape[0]
            if rows < max_length:
                pad_size = max_length - rows
                padding = np.zeros((pad_size, 300))
                matrix = np.append(matrix, padding, axis=0)
            elif rows >= max_length:
                #print "Sentence longer than " + str(max_length) + " - it has " + str(rows) + "words."
                matrix = matrix[:max_length]
            x.append(matrix)
        y = np.array(y)
        y = y.astype('int32')
        print "Shape of y for " + part + ": ", y.shape
        vecData[y_part] = y
        X = np.dstack(x)
        X = np.rollaxis(X, -1)
        X = X.reshape(num, 1, 50, 300)
        X = X.astype('float32')
        print "Shape of X for " + part + ": ", X.shape
        vecData[X_part] = X
    X_train = vecData["X_train"]
    y_train = vecData["y_train"]
    X_dev = vecData["X_dev"]
    y_dev = vecData["y_dev"]
    X_test = vecData["X_test"]
    y_test = vecData["y_test"]
    return X_train, y_train, X_dev, y_dev, X_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_cnn(input_var=None):

    input_layer = lasagne.layers.InputLayer(shape=(None, 1, max_length, num_feats))

    conv_layer = lasagne.layers.Conv2DLayer(input_layer, num_filters=50, filter_size=(8, num_feats), stride=3, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    
    #dropout1 = lasagne.layers.DropoutLayer(conv_layer)

    max_pool = lasagne.layers.MaxPool2DLayer(conv_layer, pool_size=(14, 1))

    #dropout2 = lasagne.layers.DropoutLayer(max_pool)

    fully_connected = lasagne.layers.DenseLayer(max_pool, num_units=25, nonlinearity=lasagne.nonlinearities.sigmoid)

    #dropout3 = lasagne.layers.DropoutLayer(fully_connected)

    output_layer = lasagne.layers.DenseLayer(fully_connected, num_units=4, nonlinearity=lasagne.nonlinearities.softmax)

    return conv_layer, max_pool, fully_connected, output_layer

if __name__ == "__main__":

    text_data = load_text()
    
    num_utts = {}
    for part in text_data:
        num_utts[part] = len(text_data[part])

    X_train, y_train, X_dev, y_dev, X_test, y_test = vectorize_data(text_data, num_utts)

    
    input_var = T.tensor4('inputs') 
    target_var = T.ivector('targets')
    
    conv, maxp, fc, network = build_cnn(input_var)
    #layer_weights = {conv: 0.01, maxp: 0.01, fc: 0.01, network: 0.01}
    #l2penalty = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2) * 0.01
    #l1penalty = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l1) * 0.01
    prediction = lasagne.layers.get_output(network, input_var)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()# + l1penalty + l2penalty
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001, momentum=0.9)
    test_prediction = lasagne.layers.get_output(network, input_var, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])

    # Training loop
    print "Starting training..."
    num_epochs = 100
    TRAIN_error = []
    VAL_error = []
    TEST_error = []
    VAL_acc = []
    TEST_acc = []
    for epoch in range(num_epochs):
        # full pass over the training data
        train_err = 0
        train_batches = 0
        start_time=time.time()
        # to see how the training and validation error progresses
        for batch in iterate_minibatches(X_train, y_train, 50, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        # full pass over the validation data
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_dev, y_dev, 50, shuffle=False):
            inputs, targets = batch
            err, acc, pred = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            print "Predicted values: ", pred[:10]
            print "Target values: ", targets[:10]
            val_batches += 1
        # Print results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

        # Compute and print the test error
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 50, shuffle=False):
            inputs, targets = batch
            err, acc, pred = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            print "Predicted values: ", pred[:10]
            print "Target values: ", targets[:10]
            test_batches += 1

        TRAIN_error.append(train_err/train_batches)
        VAL_error.append(val_err/val_batches)
        VAL_acc.append(val_acc/val_batches)
        TEST_error.append(test_err/test_batches)
        TEST_acc.append(test_acc/test_batches)
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f}".format(test_acc / test_batches * 100))
        
    fig = plt.figure()
    error = fig.add_subplot(211)
    error.plot(TRAIN_error, c='b', label='Training error')
    error.plot(VAL_error, c='g', label='Validation error')
    error.plot(TEST_error, c='r', label='Test error')
    error.set_xlabel('epoch')
    error.set_ylabel('error')
    error.legend(loc='upper right')

    acc = fig.add_subplot(212)
    acc.plot(VAL_acc, c='g', label='Validation accuracy')
    acc.plot(TEST_acc, c='r', label='Test accuracy')
    acc.set_xlabel('epoch')
    acc.set_ylabel('accuracy')
    acc.legend(loc='lower right')
    plt.savefig('newData_batch50_lambda001.png')

    
    print "END OF PROCESS"

