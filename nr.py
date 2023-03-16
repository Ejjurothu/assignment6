import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load dataset
dataset = pd.read_csv("breastcancer.csv", header=None).values

# split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,1:10], dataset[:,10],
                                                    test_size=0.25, random_state=87)

# normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# set random seed
np.random.seed(155)

# create model
my_first_nn = Sequential()
my_first_nn.add(Dense(20, input_dim=9, activation='relu')) # hidden layer
my_first_nn.add(Dense(23, activation='tanh')) 
my_first_nn.add(Dense(30, activation='sigmoid'))
my_first_nn.add(Dense(30, activation='softmax'))
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer

# compile model
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)

# print model summary
print(my_first_nn.summary())

# evaluate model on testing set and print accuracy
accuracy = my_first_nn.evaluate(X_test, Y_test)[1]
print("Accuracy (with normalization): {:.2f}%".format(accuracy * 100))
