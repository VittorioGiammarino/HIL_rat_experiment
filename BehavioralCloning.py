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
    
    
def MakePredictions(model, stateSpace):
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    
    deterministic_policy = np.empty(0)
    predictions = probability_model.predict(stateSpace[:,:])
    for i in range(stateSpace.shape[0]):    
        deterministic_policy = np.append(deterministic_policy, 
                                         np.argmax(predictions[i,:]))
        
    return predictions, deterministic_policy

def EvaluationNN1(map, stateSpace, P, traj, control, ntraj):
    average_NN = np.empty((0))
    success_percentageNN = np.empty((0))
    average_expert = np.empty((0))

    for i in range(len(ntraj)):
        action_space=5
        labels, TrainingSet = ProcessData(traj[0:ntraj[i]][:],control[0:ntraj[i]][:],stateSpace)
        model = NN1(action_space)
        model.fit(TrainingSet, labels, epochs=50)
        predictions, deterministic_policy = MakePredictions(model, stateSpace)
        T=100
        base=ss.BaseStateIndex(stateSpace,map)
        TERMINAL_STATE_INDEX = ss.TerminalStateIndex(stateSpace,map)
        [trajNN,controlNN,flagNN]=sim.StochasticSampleTrajMDP(P, predictions, 1000, T, base, TERMINAL_STATE_INDEX)
        length_trajNN = np.empty((0))
        for j in range(len(trajNN)):
            length_trajNN = np.append(length_trajNN, len(trajNN[j][:]))
        average_NN = np.append(average_NN,np.divide(np.sum(length_trajNN),len(length_trajNN)))
        success_percentageNN = np.append(success_percentageNN,np.divide(np.sum(flagNN),len(length_trajNN)))
    
        length_traj = np.empty((0))
        for k in range(ntraj[i]):
            length_traj = np.append(length_traj, len(traj[k][:]))

        average_expert = np.append(average_expert, np.divide(np.sum(length_traj),len(length_traj)))
    
    return average_NN, success_percentageNN, average_expert

def EvaluationNN2(map, stateSpace, P, traj, control, ntraj):
    average_NN = np.empty((0))
    success_percentageNN = np.empty((0))
    average_expert = np.empty((0))

    for i in range(len(ntraj)):
        action_space=5
        labels, TrainingSet = ProcessData(traj[1:ntraj[i]][:],control[1:ntraj[i]][:],stateSpace)
        model = NN2(action_space)
        encoded = tf.keras.utils.to_categorical(labels)
        model.fit(TrainingSet, encoded, epochs=50)
        predictions, deterministic_policy = MakePredictions(model, stateSpace)
        T=1000
        base=ss.BaseStateIndex(stateSpace,map)
        TERMINAL_STATE_INDEX = ss.TerminalStateIndex(stateSpace,map)
        [trajNN,controlNN,flagNN]=sim.StochasticSampleTrajMDP(P, predictions, 1000, T, base, TERMINAL_STATE_INDEX)
        length_trajNN = np.empty((0))
        for j in range(len(trajNN)):
            length_trajNN = np.append(length_trajNN, len(trajNN[j][:]))
        average_NN = np.append(average_NN,np.divide(np.sum(length_trajNN),len(length_trajNN)))
        success_percentageNN = np.append(success_percentageNN,np.divide(np.sum(flagNN),len(length_trajNN)))
    
        length_traj = np.empty((0))
        for k in range(ntraj[i]):
            length_traj = np.append(length_traj, len(traj[k][:]))

        average_expert = np.append(average_expert, np.divide(np.sum(length_traj),len(length_traj)))
    
    return average_NN, success_percentageNN, average_expert

def EvaluationNN3(map, stateSpace, P, traj, control, ntraj):
    average_NN = np.empty((0))
    success_percentageNN = np.empty((0))
    average_expert = np.empty((0))

    for i in range(len(ntraj)):
        action_space=5
        labels, TrainingSet = ProcessData(traj[1:ntraj[i]][:],control[1:ntraj[i]][:],stateSpace)
        model = NN3(action_space)
        encoded = tf.keras.utils.to_categorical(labels)
        model.fit(TrainingSet, encoded, epochs=50)
        predictions, deterministic_policy = MakePredictions(model, stateSpace)
        T=1000
        base=ss.BaseStateIndex(stateSpace,map)
        TERMINAL_STATE_INDEX = ss.TerminalStateIndex(stateSpace,map)
        [trajNN,controlNN,flagNN]=sim.StochasticSampleTrajMDP(P, predictions, 1000, T, base, TERMINAL_STATE_INDEX)
        length_trajNN = np.empty((0))
        for j in range(len(trajNN)):
            length_trajNN = np.append(length_trajNN, len(trajNN[j][:]))
        average_NN = np.append(average_NN,np.divide(np.sum(length_trajNN),len(length_trajNN)))
        success_percentageNN = np.append(success_percentageNN,np.divide(np.sum(flagNN),len(length_trajNN)))
    
        length_traj = np.empty((0))
        for k in range(ntraj[i]):
            length_traj = np.append(length_traj, len(traj[k][:]))

        average_expert = np.append(average_expert, np.divide(np.sum(length_traj),len(length_traj)))
    
    return average_NN, success_percentageNN, average_expert