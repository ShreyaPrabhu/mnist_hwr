import numpy as np
import math
import sys
import random
import time

def initialize_numpy_matrix(row, col, value):
    # He-et-al Initialization for weights
    # formula - sqrt(2/size)
    return np.random.randn(row, col) * math.sqrt(2.0 / value)

def get_weights_biases():
    weights = {}
    weights["weights_input_hidden1"] = initialize_numpy_matrix(550, 784, 784)
    weights["weights_hidden1_hidden2"] = initialize_numpy_matrix(250, 550, 550)
    weights["weights_hidden2_output"] = initialize_numpy_matrix(10, 250, 250)
    biases = {}
    biases["bias_hidden1"] = initialize_numpy_matrix(550, 1, 784)
    biases["bias_hidden2"] = initialize_numpy_matrix(250, 1, 550)
    biases["bias_output"] = initialize_numpy_matrix(10, 1, 250)
    return weights, biases

def update_weight_biases(weights, biases, backward_pass_data):
    learning_rate = 0.01
    weights['weights_input_hidden1'] = weights['weights_input_hidden1'] - (learning_rate * backward_pass_data['updated_weights_input_hidden1'])
    weights['weights_hidden1_hidden2'] = weights['weights_hidden1_hidden2'] - (learning_rate * backward_pass_data['updated_weights_hidden1_hidden2'])
    weights['weights_hidden2_output'] = weights['weights_hidden2_output'] - (learning_rate * backward_pass_data['updated_weights_hidden2_output'])
    biases['bias_hidden1'] = biases['bias_hidden1'] - (learning_rate * backward_pass_data['updated_bias_hidden1'])
    biases['bias_hidden2'] = biases['bias_hidden2'] - (learning_rate * backward_pass_data['updated_bias_hidden2'])
    biases['bias_output'] = biases['bias_output'] - (learning_rate * backward_pass_data['updated_bias_output'])
    return weights, biases

def read_input_images(train_data_file, train_label_file):
    train_data = np.loadtxt(open(train_data_file, "rb"), delimiter=",")
    train_label = np.loadtxt(open(train_label_file, "rb"), delimiter=',')
    # sample_idx = np.random.randint(0, train_data.shape[0], 10000)
    # train_data = train_data[sample_idx]
    # train_label = train_label[sample_idx]
    return train_data.astype(int).transpose(), train_label.astype(int).transpose()

def read_test_images(test_data_file):
    test_data = np.loadtxt(open(test_data_file, "rb"), delimiter=",")
    return test_data.astype(int).transpose()

def sigmoid_function(data):
    return 1.0/(1.0 + np.exp(-np.clip(data, -400, 400, out = data)))

def sigmoid_derivative_function(layer_activation, layer_updated_activation):
    return layer_updated_activation * layer_activation * (1 - layer_activation)

def softmax_function(data):
    e_data = np.exp(data - data.max())
    return (e_data/np.sum(e_data, axis = 0))

def one_hot_encoding(labels):
    one_hot_encoding = np.zeros((labels.size, 10))
    one_hot_encoding[np.arange(labels.size), labels] = 1
    return one_hot_encoding.transpose()

def forward_propagate(train_data, train_label, weights, biases, test):
    hidden1_values = weights['weights_input_hidden1'].dot(train_data) + biases['bias_hidden1']
    hidden1_activation = sigmoid_function(hidden1_values)
    hidden2_values = weights['weights_hidden1_hidden2'].dot(hidden1_activation) + biases['bias_hidden2']
    hidden2_activation = sigmoid_function(hidden2_values)
    output_values = weights['weights_hidden2_output'].dot(hidden2_activation) + biases['bias_output']
    output_activation = softmax_function(output_values)

    if test:
        return output_activation

    forward_pass_data = {}
    forward_pass_data["train_data"] = train_data
    forward_pass_data["hidden1_values"] = hidden1_values
    forward_pass_data["hidden1_activation"] = hidden1_activation
    forward_pass_data["hidden2_values"] = hidden2_values
    forward_pass_data["hidden2_activation"] = hidden2_activation
    forward_pass_data["output_values"] = output_values
    forward_pass_data["output_activation"] = output_activation
    forward_pass_data["label_encoded"] = train_label
    return forward_pass_data

def backward_propagate(forward_pass_data, weights, biases):
    train_data_datasize = forward_pass_data["train_data"].shape[1]
    error_output_layer = forward_pass_data["output_activation"] - forward_pass_data["label_encoded"]
    updated_weights_hidden2_output = error_output_layer.dot(forward_pass_data["hidden2_activation"].transpose())
    updated_weights_hidden2_output /= train_data_datasize
    updated_bias_output = np.sum(error_output_layer, axis = 1, keepdims = True) / train_data_datasize
    updated_hidden2_activation = weights["weights_hidden2_output"].transpose().dot(error_output_layer)
    error_hidden_layer2 = sigmoid_derivative_function(forward_pass_data['hidden2_activation'], updated_hidden2_activation)
    updated_weights_hidden1_hidden2 = error_hidden_layer2.dot(forward_pass_data['hidden1_activation'].transpose())
    updated_weights_hidden1_hidden2 /= train_data_datasize
    updated_bias_hidden2 = np.sum(error_hidden_layer2, axis = 1, keepdims = True) / train_data_datasize
    updated_hidden1_activation = weights['weights_hidden1_hidden2'].transpose().dot(error_hidden_layer2)
    error_hidden_layer1 = sigmoid_derivative_function(forward_pass_data['hidden1_activation'], updated_hidden1_activation)
    updated_weights_input_hidden1 = error_hidden_layer1.dot(forward_pass_data["train_data"].transpose())
    updated_weights_input_hidden1 /= train_data_datasize
    updated_bias_hidden1 = np.sum(error_hidden_layer1, axis = 1, keepdims = True) / train_data_datasize

    backward_pass_data = {}
    backward_pass_data["updated_weights_hidden2_output"] = updated_weights_hidden2_output
    backward_pass_data["updated_bias_output"] = updated_bias_output
    backward_pass_data["updated_weights_hidden1_hidden2"] = updated_weights_hidden1_hidden2
    backward_pass_data["updated_bias_hidden2"] = updated_bias_hidden2
    backward_pass_data["updated_weights_input_hidden1"] = updated_weights_input_hidden1
    backward_pass_data["updated_bias_hidden1"] = updated_bias_hidden1
    return backward_pass_data

def forward_backward_propagate(train_data, train_label, weights, biases):
    forward_pass = forward_propagate(train_data, train_label, weights, biases, False)
    backward_pass =  backward_propagate(forward_pass, weights, biases)
    return update_weight_biases(weights, biases, backward_pass)

def train(train_data_file, train_label_file):
    epochs = 100
    batch_size = 25
    train_data, train_label = read_input_images(train_data_file, train_label_file)
    train_label = one_hot_encoding(train_label)
    weights, biases = get_weights_biases()
    train_data_size = train_data.shape[1]
    iterations = math.floor(train_data_size/batch_size)
    batches = []
    for iter in range(0, iterations):
        batches.append((train_data[:, batch_size * iter : batch_size * (iter + 1)], train_label[:, batch_size * iter : batch_size * (iter + 1)]))
    if train_data_size % batch_size != 0:
        batches.append((train_data[:, batch_size * iterations : train_data_size], train_label[:, batch_size * iterations : train_data_size]))

    for iterations in range(0, epochs):
        for batch_train_data, batch_train_label in batches:
            weights, biases = forward_backward_propagate(batch_train_data, batch_train_label, weights, biases)
    return weights, biases

def neural_networks():
    start_time = time.time()
    train_data_file = sys.argv[1]
    train_label_file = sys.argv[2]
    test_data_file = sys.argv[3]
    weights, biases = train(train_data_file, train_label_file)
    test_data = read_test_images(test_data_file)
    test_predictions = forward_propagate(test_data, None, weights, biases, True)
    np.savetxt("test_predictions.csv", np.argmax(test_predictions, axis = 0), fmt = "%d")
    print("--- %s seconds ---" % (time.time() - start_time))

neural_networks()
