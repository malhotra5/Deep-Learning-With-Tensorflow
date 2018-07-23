import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import tensorflow as tf

#Resets the graph if it exists
tf.reset_default_graph()

#Load the mnist hand written digits dataset 
X, Y, test_x, test_y = mnist.load_data(one_hot = True)

#Reshape dataset to pass through model
X = X.reshape([-1,28,28,1])
test_x = test_x.reshape([-1,28,28,1])

#Make the convolutional layers
convnet = input_data(shape=[None,28,28,1],name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet,2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

#Make the fully connected layer
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.9)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

#Create modal, tensorboard for debugging and analysis
model = tflearn.DNN(convnet,tensorboard_dir='log')

#Fit training set
model.fit({'input': X}, {'targets': Y}, n_epoch=1,
          validation_set=({'input': test_x,}, {'targets': test_y}),
          snapshot_step=1000,
          show_metric = True,
          run_id = 'mnist')

#Save the model
model.save('handWrittenDigitsCNN.model')

#Load model for usage
model.load('handWrittenDigitsCNN.model')
#Print prediction for first in the dataset
print(model.predict([test_x[1]]))
























