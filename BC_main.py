#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:53:43 2020

@author: Vittorio Giammarino 
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import Environment as env
import StateSpace as ss
import DynamicProgramming as dp
import Simulation as sim
import BehavioralCloning as bc

# %% map generation 
map = env.Generate_world_subgoals_simplified()

# %% Generate State Space and Expert's Solution
stateSpace=ss.GenerateStateSpace(map)            
K = stateSpace.shape[0];
TERMINAL_STATE_INDEX = ss.TerminalStateIndex(stateSpace,map)
P = dp.ComputeTransitionProbabilityMatrix(stateSpace,map)
G = dp.ComputeStageCosts(stateSpace,map)
[J_opt_vi,u_opt_ind_vi] = dp.ValueIteration(P,G,TERMINAL_STATE_INDEX)

# %% Plot Optimal Solution
env.PlotOptimalSolution(map,stateSpace,u_opt_ind_vi, 'Figures/FiguresExpert/Expert_pickup.eps', 'Figures/FiguresExpert/Expert_dropoff.eps')

# %% Generate Expert's trajectories
T=6000 # Number of trajectories we wish to generate
base=ss.BaseStateIndex(stateSpace,map)
[traj,control,flag]=sim.SampleTrajMDP(P, u_opt_ind_vi, 1000, T, base, TERMINAL_STATE_INDEX)

# %% Simulation
env.VideoSimulation(map,stateSpace,control[1][:],traj[1][:], 'Videos/VideosExpert/Expert_video_simulation.mp4')

# %% NN Behavioral Cloning
action_space=5
labels, TrainingSet = bc.ProcessData(traj,control,stateSpace)
model1 = bc.NN1(action_space)
model2 = bc.NN2(action_space)
model3 = bc.NN3(action_space)

# train the models
model1.fit(TrainingSet, labels, epochs=10)
encoded = tf.keras.utils.to_categorical(labels)
model2.fit(TrainingSet, encoded, epochs=10)
model3.fit(TrainingSet, encoded, epochs=10)
predictionsNN1, deterministic_policyNN1 = bc.MakePredictions(model1, stateSpace)
predictionsNN2, deterministic_policyNN2 = bc.MakePredictions(model2, stateSpace)
predictionsNN3, deterministic_policyNN2 = bc.MakePredictions(model3, stateSpace)
        
env.PlotOptimalSolution(map,stateSpace,deterministic_policyNN1, 'Figures/FiguresBC/NN1_pickup.eps', 'Figures/FiguresBC/NN1_dropoff.eps')

# %% Simulation of NN 
T=1
base=ss.BaseStateIndex(stateSpace,map)
[trajNN1,controlNN1,flagNN1]=sim.StochasticSampleTrajMDP(P, predictionsNN1, 1000, T, base, TERMINAL_STATE_INDEX)
env.VideoSimulation(map,stateSpace,controlNN1[0][:],trajNN1[0][:],"Videos/VideosBC/sim_NN1.mp4")

# %% Evaluate Performance

ntraj = [10, 20, 50, 100, 200, 500, 1000]
average_NN1, success_percentageNN1, average_expert = bc.EvaluationNN1(map, stateSpace, P, traj, control, ntraj)
average_NN2, success_percentageNN2, average_expert = bc.EvaluationNN2(map, stateSpace, P, traj, control, ntraj)
average_NN3, success_percentageNN3, average_expert = bc.EvaluationNN3(map, stateSpace, P, traj, control, ntraj)

# %% plot performance 
plt.figure()
plt.subplot(211)
plt.plot(ntraj, average_NN1,'go--', label = 'Neural Network 1')
plt.plot(ntraj, average_NN2,'rs--', label = 'Neural Network 2')
plt.plot(ntraj, average_NN3,'cp--', label = 'Neural Network 3')
plt.plot(ntraj, average_expert,'b', label = 'Expert')
plt.ylabel('Average steps to goal')
plt.subplot(212)
plt.plot(ntraj, success_percentageNN1,'go--', label = 'Nerual Network 1')
plt.plot(ntraj, success_percentageNN2,'rs--', label = 'Nerual Network 2')
plt.plot(ntraj, success_percentageNN3,'cp--', label = 'Nerual Network 3')
plt.plot(ntraj, np.ones((len(ntraj))),'b', label='Expert')
plt.xlabel('Number of Trajectories')
plt.ylabel('Percentage of success')
plt.legend(loc='lower right')
plt.savefig('Figures/FiguresBC/evaluationBehavioralCloning.eps', format='eps')
plt.show()




