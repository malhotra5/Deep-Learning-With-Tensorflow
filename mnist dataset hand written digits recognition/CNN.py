import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#Load dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#Number of output y-values
numClass = 10
#Mini-batch gradient descent parameter
batch_size = 128

#x and y placeholders
x = tf.placeholder('float',[None, 28*28])
y = tf.placeholder('float')

#Dropout rate
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

#Helper functions for convolutional layers
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#Defining model
def convolutional_neural_network_model(data):
    #Define weights and biases
    weights = {'Conv1':tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'Conv2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'Fully_connected_layer':tf.Variable(tf.random_normal([7*7*64, 1024])),
               'Out_layer':tf.Variable(tf.random_normal([1024, numClass]))}
    biases = {'Conv1':tf.Variable(tf.random_normal([32])),
              'Conv2':tf.Variable(tf.random_normal([64])),
              'Fully_connected_layer':tf.Variable(tf.random_normal([1024])),
              'Out_layer':tf.Variable(tf.random_normal([numClass]))}

    #Reshape the data
    x = tf.reshape(data, shape=[-1,28,28,1])

    #Convolutional layers
    conv1 = tf.nn.relu(conv2d(x, tf.add(weights['Conv1'], biases['Conv1'])))
    conv1 = maxpool2d(conv1)    
    conv2 = tf.nn.relu(conv2d(conv1, tf.add(weights['Conv2'], biases['Conv2'])))
    conv2 = maxpool2d(conv2)

    #Fully connected layer
    fully_connected_layer = tf.reshape(conv2,[-1,7*7*64])
    fully_connected_layer = tf.add(tf.matmul(fully_connected_layer, weights['Fully_connected_layer']), biases['Fully_connected_layer'])
    fully_connected_layer = tf.nn.relu(fully_connected_layer)

    #Dropout
    fully_connected_layer = tf.nn.dropout(fully_connected_layer, keep_rate)

    #Final Output
    output = tf.add(tf.matmul(fully_connected_layer, weights['Out_layer']), biases['Out_layer'])
    output = tf.nn.relu(output)
    
    return output
    

def train_neural_network(x):
    #Prediction
    prediction = convolutional_neural_network_model(x)
    #Optimizer for cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #Number of epochs
    numEpochs = 5
    #Initializer
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        #Initializing variables
        sess.run(init)

        #Running epochs
        for epoch in range(numEpochs):
            #Debugging values
            epoch_cost = 0

            #Minimizing cost in batches
            for _ in range(int(mnist.train.num_examples/batch_size)):
                batch_x , batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_cost = epoch_cost + c
            print('Epoch ' + str(epoch+1) + ' out of ' + str(numEpochs) + " epoch loss: " + str(epoch_cost))

        #Test set accuracy results
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        f1 = accuracy.eval({x: mnist.test.images, y:mnist.test.labels})
        print('Accuracy: ' + str(f1))

#Call for training network
train_neural_network(x)
