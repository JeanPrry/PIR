# Inspired from the youtube channel Machine Learnia

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm   # We import the tqdm library to visualize the progress bar


plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor":  (0.12 , 0.12, 0.12, 1),
    "axes.facecolor": (0.12 , 0.12, 0.12, 1),
})


def init(dims):   # Randomly initialize the weights and biases of the model where dims is a list containing the number of neurons for each layer    

    params = {}
    for l in range(1, len(dims)):   # We iterate over the layers
        params['W' + str(l)] = np.random.randn(dims[l], dims[l-1])  # We initialize the weights  
        params['b' + str(l)] = np.random.randn(dims[l], 1)  # We initialize the biases
        
    return params   # We return the parameters


def forward_propagation(X, params): # We define the forward propagation function
    
    activations = {'A0': X}    # We create a dictionary to store the activations

    L = len(params) // 2    # We get the number of layers
    for l in range(1, L + 1):  # We iterate over the layers
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        A = activations['A' + str(l-1)]
        Z = np.dot(W, A) + b
        
        activations['A' + str(l)] = 1 / (1 + np.exp(-Z))

    return activations   # We return the activations


def log_loss(y, A): # We define the loss function
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


def back_propagation(y, params, activations):    # We define the back propagation function

    m = y.shape[1]  # We get the number of datas
    L = len(params) // 2  # We get the number of layers
    
    dZ = activations['A' + str(L)] - y  # We calculate the gradient of the loss with respect to the output of the last layer
    grads = {}  # We create a dictionary to store the gradients

    for l in reversed(range(1, L + 1)):  # We iterate over the layers from the last to the first
        grads['dW' + str(l)] = 1 / m * np.dot(dZ, activations['A' + str(l-1)].T)  # We calculate the gradient of the loss with respect to the weights of the current layer
        grads['db' + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)  # We calculate the gradient of the loss with respect to the bias of the current layer
        if l > 1:  # We calculate the gradient of the loss with respect to the output of the previous layer only if we are not in the first layer
            dZ = np.dot(params['W' + str(l)].T, dZ) * (activations['A' + str(l-1)] * (1 - activations['A' + str(l-1)]))  # We calculate the gradient of the loss with respect to the output of the previous layer
    
    return grads    # We return the gradients


def update(params, grads, lr):  # We define the update function

    L = len(params) // 2  # We get the number of layers
    
    for l in range(1, L + 1):   # We iterate over the layers
        params['W' + str(l)] -= lr * grads['dW' + str(l)]  # We update the weights of the current layer
        params['b' + str(l)] -= lr * grads['db' + str(l)]  # We update the bias of the current layer

    return params   # We return the parameters


def predict(X, params):   # We define the predict function
    activations = forward_propagation(X, params)    # We get the activations
    return activations['A' + str(len(params) // 2)] >= 0.5 # We return the prediction


def neural_network(X, y, hidden_layers, lr, epochs):

    dims = list(hidden_layers)   # We get the number of neurons for each layer
    dims.insert(0, X.shape[0])  # We add the number of neurons of the input layer
    dims.append(y.shape[0]) # We add the number of neurons of the output layer

    params = init(dims) # We initialize the parameters

    L = len(params) // 2    # We get the number of layers

    training_history = np.zeros((int(epochs), 2))    # We create a matrix to store the loss and the accuracy for each epoch    

    for epoch in tqdm(range(epochs)):  # We iterate over the epochs

        activations = forward_propagation(X, params)    # We get the activations
        grads = back_propagation(y, params, activations)  # We get the gradients
        params = update(params, grads, lr)  # We update the parameters
        Af = activations['A' + str(L)]  # We get the output of the last layer
        training_history[epoch, 0] = (log_loss(y.flatten(), Af.flatten()))  # We store the loss for the current epoch
        y_pred = predict(X, params) # We get the prediction
        training_history[epoch, 1] = (metrics.accuracy_score(y.flatten(), y_pred.flatten()))    # We store the accuracy for the current epoch
            

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()

    return params    # We return the parameters
