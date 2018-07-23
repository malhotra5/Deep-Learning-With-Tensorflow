import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#Read data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#Number of nodes in hidden layers
hidden_layer1_nodes = 500
hidden_layer2_nodes = 500
hidden_layer3_nodes = 500

#Number of output y-values
numClass = 10

#Mini-batch gradient descent parameter
batch_size = 100

#Place holders
x = tf.placeholder('float',[None, 28*28])
y = tf.placeholder('float')


def neural_network_model(data):
    #Define weights and biases
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([28*28, hidden_layer1_nodes])),
                      'biases': tf.Variable(tf.random_normal([hidden_layer1_nodes]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([hidden_layer1_nodes, hidden_layer2_nodes])),
                      'biases': tf.Variable(tf.random_normal([hidden_layer2_nodes]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([hidden_layer2_nodes,hidden_layer3_nodes])),
                       'biases': tf.Variable(tf.random_normal([hidden_layer3_nodes]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([hidden_layer3_nodes, numClass])),
                    'biases': tf.Variable(tf.random_normal([numClass]))}

    #Forward Propagation
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output
    

def train_neural_network(x):
    #Prediction and optimization for cost
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    numEpochs = 10
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        #Initialize variables
        sess.run(init)

        for epoch in range(numEpochs):
            epoch_cost = 0
            #Run optimizer in batches 
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


train_neural_network(x)
