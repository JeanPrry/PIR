from scipy.io import arff
import pandas as pd
from neural_network import *


def binary_string_to_float(s):
    # Convert binary string to a list of integers
    int_list = [ord(c) for c in s.decode('utf-8')]

    # Concatenate integers to obtain a single unique integer for the string
    unique_int = sum(int_list[i] * 256**(len(int_list)-i-1) for i in range(len(int_list)))

    # Normalize the integer to the range [0, 1) and return as a float
    unique_int = unique_int / 256**len(int_list)
    return unique_int


def encode_data(X, y):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if type(X[i, j]) == bytes:
                X[i, j] = binary_string_to_float(X[i, j])   # We encode the data
        y[0][i] = 1 if y[0][i] == b'normal' else 0

    X = X / np.max(X)   # We normalize the data
    y = y / np.max(y)   
    return X, y


nb_data = 10000

# We import the data
data = arff.loadarff('nsl-kdd/KDDTrain+.arff')

# We convert the data into a pandas dataframe
df = pd.DataFrame(data[0])

# We convert the dataframe into a numpy array
X = df.to_numpy()

X_train = X[:nb_data, :-1]  
y_train = X[:nb_data, -1]   
y_train = y_train.reshape(1, nb_data)   

X_train, y_train = encode_data(X_train, y_train) 

X_train = X_train.astype(np.float64)    # We convert the data into float64
y_train = y_train.astype(np.int64)

print('X type: ', type(X_train[0, 0]))
print('y type: ', type(y_train[0, 0]))

print('X_train dimension: ', X_train.T.shape)
print('y_train dimension: ', y_train.shape)

params = neural_network(X_train.T, y_train, hidden_layers = (16, 16, 16, 16), lr = 0.001, epochs = 1000)