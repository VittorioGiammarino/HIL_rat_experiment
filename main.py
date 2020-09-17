#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:52:47 2020

@author: vittorio
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
import numpy as np
import matplotlib.pyplot as plt
import Environment as env
import StateSpace as ss
import DynamicProgramming as dp
import Simulation as sim
import BehavioralCloning as bc
import HierarchicalImitationLearning as hil
import concurrent.futures
from joblib import Parallel, delayed
import multiprocessing

# %% map generation 
map = env.Generate_world_subgoals_simplified()

# %% Generate State Space
stateSpace=ss.GenerateStateSpace(map)            
K = stateSpace.shape[0];
TERMINAL_STATE_INDEX = ss.TerminalStateIndex(stateSpace,map)
P = dp.ComputeTransitionProbabilityMatrix(stateSpace,map)
G = dp.ComputeStageCosts(stateSpace,map)
[J_opt_vi,u_opt_ind_vi] = dp.ValueIteration(P,G,TERMINAL_STATE_INDEX)

# %% Plot Optimal Solution
env.PlotOptimalSolution(map,stateSpace,u_opt_ind_vi, 'Figures/FiguresExpert/Expert_pickup.eps', 'Figures/FiguresExpert/Expert_dropoff.eps')

# %% Generate Expert's trajectories
T=10
base=ss.BaseStateIndex(stateSpace,map)
[traj,control,flag]=sim.SampleTrajMDP(P, u_opt_ind_vi, 1000, T, base, TERMINAL_STATE_INDEX)
labels, TrainingSet = bc.ProcessData(traj,control,stateSpace)

# %% Simulation
env.VideoSimulation(map,stateSpace,control[1][:],traj[1][:], 'Videos/VideosExpert/Expert_video_simulation.mov')

# %% HIL initialization
option_space = 2
action_space = 5
termination_space = 2
size_input = TrainingSet.shape[1]

NN_options = hil.NN_options(option_space, size_input)
NN_actions = hil.NN_actions(action_space, size_input)
NN_termination = hil.NN_termination(termination_space, size_input)

N=10 #Iterations
zeta = 0.1 #Failure factor
mu = np.ones(option_space)*np.divide(1,option_space) #initial option probability distribution

gain_lambdas = np.logspace(1, 2, 2, dtype = 'float32')
gain_eta = np.logspace(1, 3, 1, dtype = 'float32')
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)

Triple = hil.Triple(NN_options, NN_actions, NN_termination)

env_specs = hil.Environment_specs(P, stateSpace, map)
max_epoch = 1000

ED = hil.Experiment_design(labels, TrainingSet, size_input, action_space, option_space, termination_space, N, zeta, mu, 
                           Triple, LAMBDAS, ETA, env_specs, max_epoch)

# %% Validation of regularizers
inputs = range(len(ETA))
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores, prefer="threads")(delayed(hil.ValidationBW_reward)(i, ED) for i in inputs)

# %% Get the best Triple
averageHIL = np.empty((0))
for j in range(len(results)):
    averageHIL = np.append(averageHIL, results[j][1])
    
Bestid = np.argmin(averageHIL) 
Best_Triple = results[Bestid][0] 

Best_Triple.save(ED.gain_lambdas[Bestid], ED.gain_eta[Bestid])

# %% Load Model
lambdas = tf.Variable(initial_value=1.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=100., trainable=False)
lambda_gain = lambdas.numpy()[0]
eta_gain = eta.numpy()
NN_options, NN_actions, NN_termination = hil.Triple.load(lambda_gain, eta_gain)
          
# %% Evaluation 
Trajs=100
base=ss.BaseStateIndex(stateSpace,map)
[trajHIL,controlHIL,OptionsHIL, 
 TerminationHIL, flagHIL]=sim.HierarchicalStochasticSampleTrajMDP(P, stateSpace, NN_options, 
                                                                  NN_actions, NN_termination, mu, 1000, 
                                                                  Trajs, base, TERMINAL_STATE_INDEX, zeta, option_space)

length_trajHIL = np.empty((0))
for j in range(len(trajHIL)):
    length_trajHIL = np.append(length_trajHIL, len(trajHIL[j][:]))
                                                                  
best = np.argmin(length_trajHIL)                                                                  
                                                                  
# %% Video of Best Simulation
env.HILVideoSimulation(map,stateSpace,controlHIL[best][:],trajHIL[best][:],OptionsHIL[0][:],"Videos/VideosHIL/sim_HIL.mp4")



