#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:40:21 2020

@author: vittorio
"""

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
import DiscoveryDeepOptions as ddo
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
T=20
base=ss.BaseStateIndex(stateSpace,map)
traj, control, psi_evolution, reward = sim.SampleTrajMDP(P, u_tot_Expert, 100, T, base, R1_STATE_INDEX,R2_STATE_INDEX)
labels, TrainingSet = bc.ProcessData(traj,control,psi_evolution,stateSpace)

# %% Simulation
env.VideoSimulation(map,stateSpace,control[1][:],traj[1][:], psi_evolution[1][:], 'Videos/VideosExpert/Expert_video_simulation.mp4')

# %% DDO initialiazation

option_space = 2
action_space = 5
termination_space = 2
size_input = TrainingSet.shape[1]
NN_low = []
NN_termination = []

NN_high = ddo.NN_options(option_space, size_input)
for options in range(option_space):
    NN_low.append(ddo.NN_actions(action_space, size_input))
    NN_termination.append(ddo.NN_termination(termination_space, size_input))

N1=10 #Iterations
N2=5
zeta = 0.0001 #Failure factor
mu = np.ones(option_space)*np.divide(1,option_space) #initial option probability distribution
# env_specs = ddo.Environment_specs(P, stateSpace, map)
# max_epoch = 300
lambda_gain = 0.1
M_step_epoch = 50
size_batch = 32
optimizer = keras.optimizers.Adamax(learning_rate=1e-3)

DiscoveryDeepOptions = ddo.DDO(labels, TrainingSet, size_input, action_space, option_space, termination_space, 
                               N1, N2, zeta, mu, NN_high, NN_low, NN_termination,lambda_gain,M_step_epoch,size_batch, optimizer)
      
# %% Baum-Welch for provable HIL iteration

NN_b, NN_lo, NN_hi = DiscoveryDeepOptions.BaumWelch()

# %% policy analysis

#pi_hi
psi = 3
input_NN = np.concatenate((stateSpace, psi*np.ones((len(stateSpace),1))),1)
Pi_HI = np.argmax(NN_hi(input_NN).numpy(),1) 
env.PlotOptimalOptions(map,stateSpace,Pi_HI, 'Figures/FiguresDDO/Pi_HI_psi{}.eps'.format(psi))


for o in range(option_space):  
    #pi_lo
    Pi_lo = np.argmax(NN_lo[o](input_NN).numpy(),1)  
    env.PlotOptimalSolution(map,stateSpace,Pi_lo, 'Figures/FiguresDDO/PI_LO_o{}_psi{}.eps'.format(o, psi))
    #pi_b
    Pi_b = np.argmax(NN_b[o](input_NN).numpy(),1) 
    env.PlotOptimalOptions(map,stateSpace,Pi_b, 'Figures/FiguresDDO/PI_b_o{}_psi{}.eps'.format(o, psi))

# %% Evaluation 
Trajs=150
base=ss.BaseStateIndex(stateSpace,map)
[trajHIL,controlHIL,OptionsHIL, 
 TerminationHIL, psiHIL, rewardHIL]=ddo.DDO.Simulation.HierarchicalStochasticSampleTrajMDP(P, stateSpace, NN_hi, 
                                                                                           NN_lo, NN_b, mu, 300, 
                                                                                           Trajs, base, R1_STATE_INDEX, R2_STATE_INDEX, 
                                                                                           zeta, option_space)
                                                                            
best = np.argmax(rewardHIL)                                                                
                                                                  
# %% Video of Best Simulation 
env.HILVideoSimulation(map,stateSpace,controlHIL[best][:],trajHIL[best][:],OptionsHIL[best][:], psiHIL[best][:],"Videos/VideosDDO/sim_DDO.mp4")

