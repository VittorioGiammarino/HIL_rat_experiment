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

# %% Generate State Space
stateSpace=ss.GenerateStateSpace(map)            
K = stateSpace.shape[0];
R2_STATE_INDEX = ss.R2StateIndex(stateSpace,map)
R1_STATE_INDEX = ss.R1StateIndex(stateSpace,map)
P = dp.ComputeTransitionProbabilityMatrix(stateSpace,map)
GR1 = dp.ComputeStageCostsR1(stateSpace,map)
GR2 = dp.ComputeStageCostsR2(stateSpace,map)
GBoth = dp.ComputeStageCostsR1andR2(stateSpace, map)
[J_opt_vi_R1,u_opt_ind_vi_R1] = dp.ValueIteration(P,GR1,R1_STATE_INDEX)
[J_opt_vi_R2,u_opt_ind_vi_R2] = dp.ValueIteration(P,GR2,R2_STATE_INDEX)
[J_opt_vi_Both,u_opt_ind_vi_Both] = dp.ValueIteration_Both(P,GBoth,R1_STATE_INDEX,R2_STATE_INDEX)
u_opt_ind_vi_R1 = u_opt_ind_vi_R1.reshape(len(u_opt_ind_vi_R1),1)
u_opt_ind_vi_R2 = u_opt_ind_vi_R2.reshape(len(u_opt_ind_vi_R2),1)
u_opt_ind_vi_Both = u_opt_ind_vi_Both.reshape(len(u_opt_ind_vi_Both),1)
u_tot_Expert = np.concatenate((u_opt_ind_vi_R1, u_opt_ind_vi_R2, u_opt_ind_vi_Both,ss.HOVER*np.ones((len(u_opt_ind_vi_R1),1))),1)
# %% Plot Optimal Solution

env.PlotOptimalSolution(map,stateSpace,u_opt_ind_vi_R1, 'Figures/FiguresExpert/Expert_R1.eps')
env.PlotOptimalSolution(map,stateSpace,u_opt_ind_vi_R2, 'Figures/FiguresExpert/Expert_R2.eps')
env.PlotOptimalSolution(map,stateSpace,u_opt_ind_vi_Both, 'Figures/FiguresExpert/Expert_Both.eps')

# %% Generate Expert's trajectories
T=150
base=ss.BaseStateIndex(stateSpace,map)
traj, control, psi_evolution, reward = sim.SampleTrajMDP(P, u_tot_Expert, 300, T, base, R1_STATE_INDEX,R2_STATE_INDEX)
labels, TrainingSet = bc.ProcessData(traj,control,psi_evolution,stateSpace)

# %% Simulation
env.VideoSimulation(map,stateSpace,control[1][:],traj[1][:], psi_evolution[1][:], 'Videos/VideosExpert/Expert_video_simulation.mp4')

# %% NN Behavioral Cloning
action_space=5
labels, TrainingSet = bc.ProcessData(traj,control,psi_evolution,stateSpace)
model1 = bc.NN1(action_space)
model2 = bc.NN2(action_space)
model3 = bc.NN3(action_space)

# train the models
model1.fit(TrainingSet, labels, epochs=10)
encoded = tf.keras.utils.to_categorical(labels)
model2.fit(TrainingSet, encoded, epochs=10)
model3.fit(TrainingSet, encoded, epochs=10)

# %% policy synthesis

predictionsNN1, deterministic_policyNN1 = bc.GetFinalPolicy(model1, stateSpace, action_space)
predictionsNN2, deterministic_policyNN2 = bc.GetFinalPolicy(model2, stateSpace, action_space)
predictionsNN3, deterministic_policyNN2 = bc.GetFinalPolicy(model2, stateSpace, action_space)
        
env.PlotOptimalSolution(map, stateSpace, deterministic_policyNN1[:,0], 'Figures/FiguresBC/NN1_R1.eps')
env.PlotOptimalSolution(map, stateSpace, deterministic_policyNN1[:,1], 'Figures/FiguresBC/NN1_R2.eps')
env.PlotOptimalSolution(map, stateSpace, deterministic_policyNN1[:,2], 'Figures/FiguresBC/NN1_Both.eps')
env.PlotOptimalSolution(map, stateSpace, deterministic_policyNN1[:,3], 'Figures/FiguresBC/NN1_None.eps')

# %% Simulation of NN 
T=1
base=ss.BaseStateIndex(stateSpace,map)
trajNN1, controlNN1, psi_evolutionNN1, rewardNN1 = sim.StochasticSampleTrajMDP(P, predictionsNN1, 100, T, base, R1_STATE_INDEX, R2_STATE_INDEX)
env.VideoSimulation(map,stateSpace,controlNN1[0][:],trajNN1[0][:], psi_evolutionNN1[0][:],"Videos/VideosBC/sim_NN1.mp4")

# %% Evaluate Performance

ntraj = [10, 20, 50, 100]
average_expert = bc.ExpertAverageReward(reward, ntraj)
average_NN1 = bc.EvaluationNN1(map, stateSpace, P, traj, control, psi_evolution, ntraj)
average_NN2 = bc.EvaluationNN2(map, stateSpace, P, traj, control, psi_evolution, ntraj)
average_NN3 = bc.EvaluationNN3(map, stateSpace, P, traj, control, psi_evolution, ntraj)


# %% plot performance 
plt.figure()
plt.subplot(111)
plt.plot(ntraj, average_NN1,'go--', label = 'Neural Network 1')
plt.plot(ntraj, average_NN2,'rs--', label = 'Neural Network 2')
plt.plot(ntraj, average_NN3,'cp--', label = 'Neural Network 3')
plt.plot(ntraj, average_expert,'b', label = 'Expert')
plt.ylabel('Average reward')
plt.xlabel('Number of Trajectories')
plt.legend(loc='middle right')
plt.savefig('Figures/FiguresBC/evaluationBehavioralCloning.eps', format='eps')
plt.show()




