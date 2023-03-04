from scipy.io import arff
import pandas as pd
import numpy as np
from neural_network import neural_network

nb_data = 1000

# We import the data
data = arff.loadarff('nsl-kdd/KDDTrain+.arff')

# We convert the data into a pandas dataframe
df = pd.DataFrame(data[0])

# We convert the dataframe into a numpy array
X = df.to_numpy()

X_train = X[:nb_data, :-1]  
y_train = X[:nb_data, -1]
y_train = y_train.reshape(1, nb_data)   

print('X_train dimension: ', X_train.T.shape)
print('y_train dimension: ', y_train.shape)

params = neural_network(X_train, y_train, hidden_layers = (4, 4), lr = 0.01, epochs = 1000)