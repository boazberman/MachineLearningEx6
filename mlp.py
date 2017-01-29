import numpy as np

def init_mlp(inputs, targets, nhidden):
    """ Initialize network """

    # Set up network size
    nin = np.shape(inputs)[1]
    nout = np.shape(targets)[1]
    ndata = np.shape(inputs)[0]
    nhidden = nhidden

    #Initialize network
    weights1 = (np.random.rand(nin+1, nhidden)-0.5)*2/np.sqrt(nin)
    weights2 = (np.random.rand(nhidden+1, nout)-0.5)*2/np.sqrt(nhidden)

    return weights1, weights2


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

    #**************************YOUR CODE HERE*********************
    #*************************************************************
    #Write the backpropagation algorithm to find the update values for weights1 and weights2.


    #*************************************************************
    #*************************************************************

    return loss, weights1_gradient, weights2_gradient, activations[-1]


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

    #**************************YOUR CODE HERE*********************
    #*************************************************************


    #*************************************************************
    #*************************************************************


    return weighted_outputs, activations


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

    #**************************YOUR CODE HERE*********************
    #*************************************************************


    #*************************************************************
    #*************************************************************

    return 0


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
    inputs = np.concatenate((inputs,np.ones((ndata,1))),axis=1)

    for n in range(nepochs):
        epoch_loss = 0
        predictions = []
        for ex_idx in range(len(inputs)):
            x = inputs[ex_idx]
            y = targets[ex_idx]
            
            # compute gradients and update the mlp
            loss, weights1_gradient, weights2_gradient, y_hat= loss_and_gradients(x, y, weights1, weights2)
            weights1 -= eta * weights1_gradient
            weights2 -= eta * weights2_gradient
            epoch_loss += loss
            predictions.append(y_hat)

        if (np.mod(n,100)==0):
            print n, epoch_loss, accuracy_on_dataset(inputs, targets, weights1, weights2)

    return weights1, weights2
        
        



