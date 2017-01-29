import numpy as np
import math


def init_mlp(inputs, targets, nhidden):
    """ Initialize network """

    # Set up network size
    nin = np.shape(inputs)[1]
    nout = np.shape(targets)[1]
    ndata = np.shape(inputs)[0]
    nhidden = nhidden

    # Initialize network
    weights1 = (np.random.rand(nin + 1, nhidden) - 0.5) * 2 / np.sqrt(nin)
    weights1 = np.array([[0.24089566, 0.27497273], [0.11917517, -0.0307281], [-0.39397026, 0.60853348]])
    weights2 = (np.random.rand(nhidden + 1, nout) - 0.5) * 2 / np.sqrt(nhidden)
    weights2 = np.array([[-0.43097], [-0.14942569], [0.04136656]])
    return weights1, weights2


def sum_of_squares_error_function(expected_output_y, actual_y):
    return 0.5 * sum(np.power(actual_y[i] - expected_output_y[i], 2) for i in xrange(len(expected_output_y)));


def loss_and_gradients(input_x, expected_output_y, weights1, weights2):
    """compute loss and gradients for a given x,y
    
    this function gets an (x,y) pair as input along with the weights of the mlp,
    computes the loss on the given (x,y), computes the gradients for each weights layer,
    and returns a tuple of loss, weights 1 gradient, weights 2 gradient.
    The loss should be calculated according to the loss function presented in the assignment
    
    Arguments:
        input_x {numpy 1d array} -- an instance from the dataset
        expected_output_y {scalar} -- the ground truth
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer
    
    Returns:
        tuple -- loss, weights 1 gradient, weights 2 gradient, and activations[-1] which is y_hat
    """
    # Initialize gradients
    weights1_gradient, weights2_gradient = np.zeros(weights1.shape), np.zeros(weights2.shape)
    # Initialize loss
    loss = 0
    weighted_outputs, activations = mlpfwd(input_x, weights1, weights2)

    # **************************YOUR CODE HERE*********************
    # *************************************************************
    # Write the backpropagation algorithm to find the update values for weights1 and weights2.
    actual_y = activations[-1]
    loss = sum_of_squares_error_function(expected_output_y, actual_y)
    weights2_gradient = np.zeros(weights2.shape)
    weights2_error = np.zeros(weights2.shape)
    for i, output_neuron in enumerate(expected_output_y):
        error = (actual_y[i] - output_neuron) * actual_y[i] * (1 - actual_y[i])
        for j, weights_to_output_neuron in enumerate(with_bias(activations[0])):
            weights2_error[j][i] += weights_to_output_neuron
            weights2_gradient[j][i] += weights_to_output_neuron * error

    weights1_gradient = np.zeros(weights1.shape)
    # for i, hidden_weight in enumerate(activations[1]):
    for i, sss in enumerate(activations[0]):
        error = sss * (1 - sss) * sum(weights2_error[i] * weights2[i])
        for j, xj in enumerate(input_x):
            weights1_gradient[j][i] += xj * error
    # print "OMG"
        # for j, weights_to_output_neuron in enumerate(input_x):
        #     weights1_gradient[j][i] += weights_to_output_neuron * error
    # weights2_gradient = [(actual_y[i] - expected_output_y[i]) * actual_y[i] * (1 - actual_y[i]) for i in
    #                      xrange(len(expected_output_y))]
    # weights1_gradient = [weights1[i]]
    
    # *************************************************************
    # *************************************************************

    return loss, weights1_gradient, weights2_gradient, actual_y


def mlpfwd(input_x, weights1, weights2):
    """feed forward
    
    this function gets an input x and feeds it through the mlp.
    
    Arguments:
        input_x {numpy 1d array} -- an instance from the dataset
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer
    
    Returns:
        tuple -- list of weighted outputs along the way, list of activations along the way:
        
        1) The first part of the tuple consists of a list, where every item in the list
        holds the values of a layer in the network, before the activation function has been applied
        on it. The value of a layer in the network is the weighted sum of the layer before it.
        
        2) The second part of the tuple consists of a list, where every item in the list holds
        the values of a layer in the network, after the activation function has been applied on it.
        Don't forget to add the bias to a layer, when required.
    """

    weighted_outputs, activations = [], []

    # **************************YOUR CODE HERE*********************
    # *************************************************************

    layer_input = input_x
    layer_output = np.zeros(weights1.shape[1])
    for neuron_index, neuron_input_weights in enumerate(weights1.T):
        for i, weight in enumerate(neuron_input_weights):
            layer_output[neuron_index] += layer_input[i] * weight
            # hidden_layer_input = with_bias(activations[-1])
    activations.append(np.array([sigmoid(output) for output in layer_output]))
    weighted_outputs.append(layer_output)

    layer_input = with_bias(activations[0])
    layer_output = np.zeros(weights2.shape[1])
    for neuron_index, neuron_input_weights in enumerate(weights2.T):
        for i, weight in enumerate(neuron_input_weights):
            layer_output[neuron_index] += layer_input[i] * weight
    activations.append(np.array([sigmoid(output) for output in layer_output]))
    weighted_outputs.append(layer_output)
    # *************************************************************
    # *************************************************************


    return weighted_outputs, activations


def with_bias(input):
    return np.append(input, np.ones(1))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def accuracy_on_dataset(inputs, targets, weights1, weights2):
    """compute accuracy
    
    this function gets a dataset and returns model's accuracy on the dataset.
    The accuracy is calculated using a threshold of 0.5:
    if the prediction is >= 0.5 => y_hat = 1
    if the prediction is < 0.5 => y_hat = 0
    
    Arguments:
        inputs {numpy 2d array} -- instances
        targets {numpy 2d array} -- ground truths
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer

    Returns:
        scalar -- accuracy on the given dataset
    """

    # **************************YOUR CODE HERE*********************
    # *************************************************************
    total = 0
    correct = 0
    for i, input_x in enumerate(inputs):
        _, activations = mlpfwd(input_x, weights1, weights2)
        expected_output = targets[i]
        actual_output = activations[-1]
        # expected_output - actual_output
        for j, expected_j_output in enumerate(expected_output):
            total += 1
            if expected_j_output == round(actual_output[j]):
                correct += 1

        # *************************************************************
    # *************************************************************

    return float(correct) / total


def mlptrain(inputs, targets, eta, nepochs, weights1, weights2):
    """train the model
    
    Arguments:
        inputs {numpy 2d array} -- instances
        targets {numpy 2d array} -- ground truths
        eta {scalar} -- learning rate
        nepochs {scalar} -- number of epochs
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer
    """
    ndata = np.shape(inputs)[0]
    # Add the inputs that match the bias node
    inputs = np.concatenate((inputs, np.ones((ndata, 1))), axis=1)

    for n in range(nepochs):
        epoch_loss = 0
        predictions = []
        for ex_idx in range(len(inputs)):
            x = inputs[ex_idx]
            y = targets[ex_idx]

            # compute gradients and update the mlp
            loss, weights1_gradient, weights2_gradient, y_hat = loss_and_gradients(x, y, weights1, weights2)
            weights1 -= eta * weights1_gradient
            weights2 -= eta * weights2_gradient
            epoch_loss += loss
            predictions.append(y_hat)

        if (np.mod(n, 100) == 0):
            print n, epoch_loss, accuracy_on_dataset(inputs, targets, weights1, weights2)

    return weights1, weights2
