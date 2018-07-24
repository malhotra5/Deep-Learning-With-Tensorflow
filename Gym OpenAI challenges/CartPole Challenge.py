#This is not a completed project and is still under work

#Helper modules
import gym
import random
import numpy as np
from statistics import mean, median
from collections import Counter
import os
from time import sleep

#Machine learning modules
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

print('import are done')

learning_rate = 0.001
env = gym.make('CartPole-v0')
minScore = 70
initGames = 2000
steps = 500
numRepeatTrains = 5

#Reseting graphs incase error was made in my program
tf.reset_default_graph()
print('reset graph')

def makeDataSet(observations, actions, dataset):
    #Making a dataset by oneHot y values
    for num,data in enumerate(actions):
        if(data == 1):
            output = [0, 1]
        else:
            output = [1, 0]
        dataset.append([observations[num], output])
    return dataset
def runGame_and_getData(model=False):
    
    if not model:
        trainDataset = [] #Needs observations made from actions
        gameScore = 0 #Score made in a game
        gameObservations = []
        gameActions = []
        successGameScores = []

        #Multiple games
        for newGames in range(initGames):
            env.reset()
            #Multiple steps
            gameScore = 0
            for step in range(steps):
                #env.render() #Only render to see final results
                action = random.randrange(0,2)
                #Results from a step 
                observation, reward, done, info = env.step(action)

                #Recording valid observations
                if(len(observation)>0):
                    #Calculate score
                    gameScore = gameScore + reward
                    gameObservations.append(observation)
                    gameActions.append(action)

                #Break if game has ended
                if(done):
                    break
            #Seeing whether data in game is valid
            if(gameScore>=minScore):
                trainDataset = makeDataSet(gameObservations, gameActions, trainDataset)
                successGameScores.append(gameScore)

        print(Counter(successGameScores))
        if(len(successGameScores)>0):
            print(mean(successGameScores))
        finalData = np.array(trainDataset)
        np.save('DataSet.npy', finalData)
        
        
        
    else:
        existingData = np.load('DataSet.npy')
        newTrainDataset = []
        gameScore = 0
        gameObservations = []
        gameActions = []
        successGameScores = []
        for games in range(initGames):
            env.reset()
            #Re-initialize variables
            gameScore = 0
            for step in range(steps):
                env.render()
                #First random guess to start the game
                if(len(gameObservations) == 0):
                    action = random.randrange(0,2)
                else:
                    action = np.argmax(model.predict(observation.reshape(-1,len(observation), 1))[0])
                observation, reward, done, info = env.step(action)

                #Check if observation is valid
                if(len(observation) >0):
                    #Calculate score
                    gameScore = gameScore + reward
                    gameObservations.append(observation)
                    gameActions.append(action)

                if(done):
                    break
            if(gameScore >= 40):
                newTrainDataset = makeDataSet(observations, actions, dataset)
                successGameScores.append(gameScore)

        
        print(mean(successGameScores))

        print(np.shape(existingData))
        print('Choice 1:' +str(gameActions.count(1)/ len(gameActions)), 'Choice 0:' + str(gameActions.count(0)/len(gameActions)))
        existingData = existingData.tolist()
        newTrainDataset = existingData+newTrainDataset
        finalNewData = np.array(newTrainDataset)
        print(np.shape(finalNewData))
        np.save('DataSet.npy', finalNewData)


def neural_network_model(input_size):
    network = input_data(shape = [None, input_size, 1], name='input')

    network = fully_connected(network,128,activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network,256,activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network,512,activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network , 2, activation='softmax')

    network - regression(network, optimizer= 'Adam', learning_rate = learning_rate, loss ='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='tmp/logs')

    return model

def train_network(training_data, model=False):
    X = []
    y = []
    for i in training_data:
        X.append([i[0]])
        y.append(i[1])
    
    X = np.array(X).reshape(-1, len(training_data[0][0]), 1)
    print(np.shape(y))
    if not model:
        model = neural_network_model(input_size = len(X[0]))
        model.fit(X, y, n_epoch = 3, snapshot_step = 500, show_metric = True, run_id='cartpole')
    else:
        model.load('player.tfl')
        tf.reset_default_graph()
        model.fit(X, y, n_epoch = 1, snapshot_step = 500, show_metric = True, run_id='cartpole')

    model.save('player.tfl')

    return model

#Reinforcement learning
def runFinalMainObjective():
    runGame_and_getData()
    trainData = np.load('DataSet.npy')
    model = train_network(trainData)
    print('done first checkpoint')
    for i in range(numRepeatTrains):
        runGame_and_getData(model)
        newData = np.load('DataSet.npy')
        model = train_network(newData)
        print('Reinforcement: ' + str(i))

runFinalMainObjective()



























