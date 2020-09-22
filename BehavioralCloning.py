#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:34:50 2020

@author: vittorio
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import StateSpace as ss
import Simulation as sim


def ProcessData(traj,control,psi,stateSpace):
    Xtr = np.empty((2,0),int)
    inputs = np.empty((3,0),int)

    for i in range(len(traj)):
        Xtr = np.append(Xtr, [traj[i][:-1], control[i][:]],axis=1)
        inputs = np.append(inputs, np.transpose(np.concatenate((stateSpace[traj[i][:-1],:], psi[i][:-1].reshape(len(psi[i])-1,1)),1)), axis=1) 
    
    labels = Xtr[1,:]
    TrainingSet = np.transpose(inputs) 
    
    return labels, TrainingSet

def NN1(action_space):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(3,)),
    keras.layers.Dense(action_space)
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresBC/model_plotNN1.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model
    
def NN2(action_space):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(3,)),
    keras.layers.Dense(action_space)
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresBC/model_plotNN2.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    
    return model

def NN3(action_space):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(3,)),
    keras.layers.Dense(action_space)
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresBC/model_plotNN3.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.Hinge(),
                  metrics=['accuracy'])
    
    return model
    
    
def MakePredictions(model, stateSpace, psi):
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    
    deterministic_policy = np.empty(0)
    predictions = probability_model.predict(np.concatenate((stateSpace[:,:],psi*np.ones((len(stateSpace),1))),1))
    for i in range(stateSpace.shape[0]):    
        deterministic_policy = np.append(deterministic_policy, 
                                         np.argmax(predictions[i,:]))
        
    return predictions, deterministic_policy

def GetFinalPolicy(model, stateSpace, action_space):
    predictions_R1, deterministic_policy_R1 = MakePredictions(model, stateSpace, 0)
    predictions_R2, deterministic_policy_R2 = MakePredictions(model, stateSpace, 1)
    predictions_Both, deterministic_policy_Both = MakePredictions(model, stateSpace, 2)
    predictions_None, deterministic_policy_None = MakePredictions(model, stateSpace, 3)
    
    policy = np.concatenate((deterministic_policy_R1.reshape(len(deterministic_policy_R1),1), 
                             deterministic_policy_R2.reshape(len(deterministic_policy_R2),1),
                             deterministic_policy_Both.reshape(len(deterministic_policy_Both),1),
                             deterministic_policy_None.reshape(len(deterministic_policy_None),1)),1)
    
    predictions = np.concatenate((predictions_R1.reshape(len(predictions_R1),action_space,1),
                                  predictions_R2.reshape(len(predictions_R2),action_space,1),
                                  predictions_Both.reshape(len(predictions_Both),action_space,1),
                                  predictions_None.reshape(len(predictions_None),action_space,1)),2)
        
    return predictions, policy


def EvaluationNN1(map, stateSpace, P, traj, control, psi, ntraj):

    average_reward_NN = np.empty((0))

    for i in range(len(ntraj)):
        action_space=5
        labels, TrainingSet = ProcessData(traj[0:ntraj[i]][:],control[0:ntraj[i]][:],psi[0:ntraj[i]][:],stateSpace)
        model = NN1(action_space)
        model.fit(TrainingSet, labels, epochs=50)
        predictions, deterministic_policy = GetFinalPolicy(model, stateSpace, action_space)
        T=100
        base=ss.BaseStateIndex(stateSpace,map)
        R1 = ss.R1StateIndex(stateSpace, map)
        R2 = ss.R2StateIndex(stateSpace, map)
        trajNN,controlNN,psiNN,rewardNN=sim.StochasticSampleTrajMDP(P, predictions, 300, T, base, R1, R2)
        average_reward_NN = np.append(average_reward_NN, np.divide(np.sum(rewardNN),len(rewardNN)))
    
    return average_reward_NN

def EvaluationNN2(map, stateSpace, P, traj, control, psi, ntraj):

    average_reward_NN = np.empty((0))

    for i in range(len(ntraj)):
        action_space=5
        labels, TrainingSet = ProcessData(traj[0:ntraj[i]][:],control[0:ntraj[i]][:],psi[0:ntraj[i]][:],stateSpace)
        model = NN2(action_space)
        encoded = tf.keras.utils.to_categorical(labels)
        model.fit(TrainingSet, encoded, epochs=50)
        predictions, deterministic_policy = GetFinalPolicy(model, stateSpace, action_space)
        T=100
        base=ss.BaseStateIndex(stateSpace,map)
        R1 = ss.R1StateIndex(stateSpace, map)
        R2 = ss.R2StateIndex(stateSpace, map)
        trajNN,controlNN,psiNN,rewardNN=sim.StochasticSampleTrajMDP(P, predictions, 300, T, base, R1, R2)
        average_reward_NN = np.append(average_reward_NN, np.divide(np.sum(rewardNN),len(rewardNN)))
    
    return average_reward_NN

def EvaluationNN3(map, stateSpace, P, traj, control, psi, ntraj):

    average_reward_NN = np.empty((0))

    for i in range(len(ntraj)):
        action_space=5
        labels, TrainingSet = ProcessData(traj[0:ntraj[i]][:],control[0:ntraj[i]][:],psi[0:ntraj[i]][:],stateSpace)
        model = NN3(action_space)
        encoded = tf.keras.utils.to_categorical(labels)
        model.fit(TrainingSet, encoded, epochs=50)
        predictions, deterministic_policy = GetFinalPolicy(model, stateSpace, action_space)
        T=100
        base=ss.BaseStateIndex(stateSpace,map)
        R1 = ss.R1StateIndex(stateSpace, map)
        R2 = ss.R2StateIndex(stateSpace, map)
        trajNN,controlNN,psiNN,rewardNN=sim.StochasticSampleTrajMDP(P, predictions, 300, T, base, R1, R2)
        average_reward_NN = np.append(average_reward_NN, np.divide(np.sum(rewardNN),len(rewardNN)))
    
    return average_reward_NN

def ExpertAverageReward(reward, ntraj):
    average_reward = np.empty((0))
    
    for i in range(len(ntraj)):
        average_reward = np.append(average_reward, np.divide(np.sum(reward[0:ntraj[i]]),ntraj[i]))
        
    return average_reward
    
    
    
    
    
    
    