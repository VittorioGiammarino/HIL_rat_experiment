#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:36:46 2020

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
T=2
base=ss.BaseStateIndex(stateSpace,map)
traj, control, psi_evolution, reward = sim.SampleTrajMDP(P, u_tot_Expert, 100, T, base, R1_STATE_INDEX,R2_STATE_INDEX)
labels, TrainingSet = bc.ProcessData(traj,control,psi_evolution,stateSpace)

# %% Simulation
env.VideoSimulation(map,stateSpace,control[1][:],traj[1][:], psi_evolution[1][:], 'Videos/VideosExpert/Expert_video_simulation.mp4')

# %% HIL initialization
option_space = 3
action_space = 5
termination_space = 2
size_input = TrainingSet.shape[1]

NN_options = hil.NN_options(option_space, size_input)
NN_actions = hil.NN_actions(action_space, size_input)
NN_termination = hil.NN_termination(termination_space, size_input)

N=10 #Iterations
zeta = 0.0001 #Failure factor
mu = np.ones(option_space)*np.divide(1,option_space) #initial option probability distribution

gain_lambdas = np.logspace(-2, 3, 3, dtype = 'float32')
gain_eta = np.logspace(-2, 3, 3, dtype = 'float32')
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)

Triple = hil.Triple(NN_options, NN_actions, NN_termination)

env_specs = hil.Environment_specs(P, stateSpace, map)

max_epoch = 300

ED = hil.Experiment_design(labels, TrainingSet, size_input, action_space, option_space, termination_space, N, zeta, mu, 
                           Triple, LAMBDAS, ETA, env_specs, max_epoch)
      
# %% Baum-Welch for provable HIL iteration

N = 10
zeta = 0.0001
mu = np.ones(option_space)*np.divide(1,option_space)
T = TrainingSet.shape[0]
TrainingSetTermination = hil.TrainingSetTermination(TrainingSet, option_space, size_input)
TrainingSetActions, labels_reshaped = hil.TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels, size_input)
lambdas = tf.Variable(initial_value=0.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=0., trainable=False)

for n in range(N):
    print('iter', n, '/', N)
    
    # Uncomment for sequential Running
    alpha = hil.Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_options, NN_actions, NN_termination)
    beta = hil.Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_options, NN_actions, NN_termination)
    gamma = hil.Gamma(TrainingSet, option_space, termination_space, alpha, beta)
    gamma_tilde = hil.GammaTilde(TrainingSet, labels, beta, alpha, 
                                  NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)
    
    
    # MultiThreading Running
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     f1 = executor.submit(hil.Alpha, TrainingSet, labels, option_space, termination_space, mu, 
    #                           zeta, NN_options, NN_actions, NN_termination)
    #     f2 = executor.submit(hil.Beta, TrainingSet, labels, option_space, termination_space, zeta, 
    #                           NN_options, NN_actions, NN_termination)  
    #     alpha = f1.result()
    #     beta = f2.result()
        
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     f3 = executor.submit(hil.Gamma, TrainingSet, option_space, termination_space, alpha, beta)
    #     f4 = executor.submit(hil.GammaTilde, TrainingSet, labels, beta, alpha, 
    #                           NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)  
    #     gamma = f3.result()
    #     gamma_tilde = f4.result()
        
    print('Expectation done')
    print('Starting maximization step')
    optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
    epochs = 30 #number of iterations for the maximization step
            
    gamma_tilde_reshaped = hil.GammaTildeReshape(gamma_tilde, option_space)
    gamma_actions_false, gamma_actions_true = hil.GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped)
    gamma_reshaped_options = hil.GammaReshapeOptions(T, option_space, gamma)
    
    
    # loss = hil.OptimizeLossAndRegularizerTot(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
    #                                          TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
    #                                          TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
    #                                          gamma, option_space, labels, size_input)
    
    loss = hil.OptimizeLossAndRegularizerTotBatch(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                                                  TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                                                  TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                                                  gamma, option_space, labels, size_input, 32)

    print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

# %% Save Model
lambda_gain = lambdas.numpy()[0]
eta_gain = eta.numpy()

Triple_model = hil.Triple(NN_options, NN_actions, NN_termination)
Triple_model.save(lambda_gain, eta_gain)

# %% Load Model

lambdas = tf.Variable(initial_value=1.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=0.1, trainable=False)
lambda_gain = lambdas.numpy()[0]
eta_gain = eta.numpy()
NN_options, NN_actions, NN_termination = hil.Triple.load(lambda_gain, eta_gain)

# %% policy analysis

#pi_hi
psi = 0
input_NN_options = np.concatenate((stateSpace, psi*np.ones((len(stateSpace),1))),1)
Pi_HI = np.argmax(NN_options(input_NN_options).numpy(),1) 
env.PlotOptimalOptions(map,stateSpace,Pi_HI, 'Figures/FiguresHIL/Pi_HI_psi{}.eps'.format(psi))

#pi_lo
o=0
input_NN_action_0 = np.concatenate((stateSpace, psi*np.ones((len(stateSpace),1)), o*np.ones((len(stateSpace),1))),1)
Pi_lo_0 = np.argmax(NN_actions(input_NN_action_0).numpy(),1) 
env.PlotOptimalSolution(map,stateSpace,Pi_lo_0, 'Figures/FiguresHIL/PI_LO_o{}_psi{}.eps'.format(o, psi))
o=1
input_NN_action_1 = np.concatenate((stateSpace, psi*np.ones((len(stateSpace),1)), o*np.ones((len(stateSpace),1))),1)
Pi_lo_1 = np.argmax(NN_actions(input_NN_action_1).numpy(),1) 
env.PlotOptimalSolution(map,stateSpace,Pi_lo_1, 'Figures/FiguresHIL/PI_LO_o{}_psi{}.eps'.format(o, psi))
o=2
input_NN_action_1 = np.concatenate((stateSpace, psi*np.ones((len(stateSpace),1)), o*np.ones((len(stateSpace),1))),1)
Pi_lo_1 = np.argmax(NN_actions(input_NN_action_1).numpy(),1) 
env.PlotOptimalSolution(map,stateSpace,Pi_lo_1, 'Figures/FiguresHIL/PI_LO_o{}_psi{}.eps'.format(o, psi))

#pi_b
o=0
input_NN_termination_0 = np.concatenate((stateSpace, psi*np.ones((len(stateSpace),1)), o*np.ones((len(stateSpace),1))),1)
Pi_b_0 = np.argmax(NN_termination(input_NN_termination_0).numpy(),1) 
env.PlotOptimalOptions(map,stateSpace,Pi_b_0, 'Figures/FiguresHIL/PI_b_o{}_psi{}.eps'.format(o, psi))
o=1
input_NN_termination_1 = np.concatenate((stateSpace, psi*np.ones((len(stateSpace),1)), o*np.ones((len(stateSpace),1))),1)
Pi_b_1 = np.argmax(NN_termination(input_NN_termination_1).numpy(),1) 
env.PlotOptimalOptions(map,stateSpace,Pi_b_1, 'Figures/FiguresHIL/PI_b_o{}_psi{}.eps'.format(o, psi))
o=2
input_NN_termination_2 = np.concatenate((stateSpace, psi*np.ones((len(stateSpace),1)), o*np.ones((len(stateSpace),1))),1)
Pi_b_2 = np.argmax(NN_termination(input_NN_termination_2).numpy(),1) 
env.PlotOptimalOptions(map,stateSpace,Pi_b_1, 'Figures/FiguresHIL/PI_b_o{}_psi{}.eps'.format(o, psi))

# %% Evaluation 
Trajs=150
base=ss.BaseStateIndex(stateSpace,map)
[trajHIL,controlHIL,OptionsHIL, 
 TerminationHIL, psiHIL, rewardHIL]=sim.HierarchicalStochasticSampleTrajMDP(P, stateSpace, NN_options, 
                                                                            NN_actions, NN_termination, mu, 300, 
                                                                            Trajs, base, R1_STATE_INDEX, R2_STATE_INDEX, 
                                                                            zeta, option_space)
                                                                            
best = np.argmax(rewardHIL)                                                                
                                                                  
# %% Video of Best Simulation 
env.HILVideoSimulation(map,stateSpace,controlHIL[best][:],trajHIL[best][:],OptionsHIL[best][:], psiHIL[best][:],"Videos/VideosHIL/sim_HIL.mp4")

# %% Evaluation on multiple trajs
ntraj = [2, 5, 10, 20, 50, 100]
averageBW, success_percentageBW, average_expert = hil.EvaluationBW(traj, control, ntraj, ED, lambdas, eta)

plt.figure()
plt.subplot(211)
plt.plot(ntraj, averageBW,'go--', label = 'HIL')
plt.plot(ntraj, average_expert,'b', label = 'Expert')
plt.ylabel('Average steps to goal')
plt.subplot(212)
plt.plot(ntraj, success_percentageBW,'go--', label = 'HIL')
plt.plot(ntraj, np.ones((len(ntraj))),'b', label='Expert')
plt.xlabel('Number of Trajectories')
plt.ylabel('Percentage of success')
plt.legend(loc='lower right')
plt.savefig('Figures/FiguresHIL/evaluationHIL_multipleTrajs.eps', format='eps')
plt.show()


    



