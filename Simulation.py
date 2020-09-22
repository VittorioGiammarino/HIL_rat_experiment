#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:10:39 2020

@author: vittorio
"""
import numpy as np

def UpdateReward(psi, x, terminal_state1, terminal_state2):
    epsilon1 = 0.9
    u1 = np.random.rand()
    epsilon2 = 0.8
    u2 = np.random.rand()
    
    if psi == 0 and u2 > epsilon2:
        psi = 2
    elif psi == 1 and u1 > epsilon1:
        psi = 2
    elif psi == 3 and u1 > epsilon1:
        psi = 0
    elif psi == 3 and u2 > epsilon2:
        psi = 1
    elif psi == 3 and u1 > epsilon1 and u2 > epsilon2:
        psi = 2
        
    if psi == 0 and x == terminal_state1:
        psi = 3
    elif psi == 1 and x == terminal_state2:
        psi = 3 
    elif psi == 2 and x == terminal_state1:
        psi = 1
    elif psi == 2 and x == terminal_state2:
        psi = 0
        
    return psi
    

def SampleTrajMDP(P, u, max_epoch, nTraj, initial_state, terminal_state1, terminal_state2):
    
    traj = [[None]*1 for _ in range(nTraj)]
    control = [[None]*1 for _ in range(nTraj)]
    reward = np.empty((0,0),int)
    psi_evolution = [[None]*1 for _ in range(nTraj)]
    
    for t in range(0,nTraj):
        
        x = np.empty((0,0),int)
        x = np.append(x, initial_state)
        u_tot = np.empty((0,0))
        psi_tot = np.empty((0,0),int)
        psi = 3
        psi_tot = np.append(psi_tot, psi)
        r=0
        
        for k in range(0,max_epoch):
            
            x_k_possible=np.where(P[x[k],:,int(u[x[k],psi])]!=0)
            prob = P[x[k],x_k_possible[0][:],int(u[x[k],psi])]
            prob_rescaled = np.divide(prob,np.amin(prob))
            
            for i in range(1,prob_rescaled.shape[0]):
                prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
            draw=np.divide(np.random.rand(),np.amin(prob))
            index_x_plus1=np.amin(np.where(draw<prob_rescaled))
            x = np.append(x, x_k_possible[0][index_x_plus1])
            u_tot = np.append(u_tot,u[x[k],psi])
            
            if x[k] == terminal_state1 or x[k] == terminal_state2:
                r = r + 1 
            
            # Randomly update the reward
            psi = UpdateReward(psi, x[k], terminal_state1, terminal_state2)
            psi_tot = np.append(psi_tot, psi)
            

        
        traj[t] = x
        control[t] = u_tot
        psi_evolution[t] = psi_tot                
        reward = np.append(reward,r)
        
    return traj, control, psi_evolution, reward


def StochasticSampleTrajMDP(P, u_policy, max_epoch, nTraj, initial_state, terminal_state1, terminal_state2):
    
    traj = [[None]*1 for _ in range(nTraj)]
    control = [[None]*1 for _ in range(nTraj)]
    reward = np.empty((0,0),int)
    psi_evolution = [[None]*1 for _ in range(nTraj)]
    
    for t in range(0,nTraj):
        
        x = np.empty((0,0),int)
        x = np.append(x, initial_state)
        u_tot = np.empty((0,0))
        psi_tot = np.empty((0,0),int)
        psi = 3
        psi_tot = np.append(psi_tot, psi)
        r=0
        
        for k in range(0,max_epoch):
            # draw action
            prob_u = u_policy[x[k],:,psi]
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[0]):
                prob_u_rescaled[i]=prob_u_rescaled[i]+prob_u_rescaled[i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            u = np.amin(np.where(draw_u<prob_u_rescaled))
            # given action, draw next state
            x_k_possible=np.where(P[x[k],:,int(u)]!=0)
            prob = P[x[k],x_k_possible[0][:],int(u)]
            prob_rescaled = np.divide(prob,np.amin(prob))
            
            for i in range(1,prob_rescaled.shape[0]):
                prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
            draw=np.divide(np.random.rand(),np.amin(prob))
            index_x_plus1=np.amin(np.where(draw<prob_rescaled))
            x = np.append(x, x_k_possible[0][index_x_plus1])
            u_tot = np.append(u_tot,u)
            
            if x[k] == terminal_state1 or x[k] == terminal_state2:
                r = r + 1 
            
            # Randomly update the reward
            psi = UpdateReward(psi, x[k], terminal_state1, terminal_state2)
            psi_tot = np.append(psi_tot, psi)
        
        traj[t] = x
        control[t] = u_tot
        psi_evolution[t] = psi_tot                
        reward = np.append(reward,r)
        
    return traj, control, psi_evolution, reward
        
def HierarchicalStochasticSampleTrajMDP(P, stateSpace, NN_options, NN_actions, NN_termination, mu, 
                                        max_epoch,nTraj,initial_state,terminal_state1, terminal_state2, zeta, option_space):
    
    traj = [[None]*1 for _ in range(nTraj)]
    control = [[None]*1 for _ in range(nTraj)]
    Option = [[None]*1 for _ in range(nTraj)]
    Termination = [[None]*1 for _ in range(nTraj)]
    reward = np.empty((0,0),int)
    psi_evolution = [[None]*1 for _ in range(nTraj)]
    
    for t in range(0,nTraj):
        
        x = np.empty((0,0),int)
        x = np.append(x, initial_state)
        u_tot = np.empty((0,0))
        o_tot = np.empty((0,0),int)
        b_tot = np.empty((0,0),int)
        psi_tot = np.empty((0,0),int)
        psi = 3
        psi_tot = np.append(psi_tot, psi)
        r=0
        
        # Initial Option
        prob_o = mu
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[0]):
            prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled))
        o_tot = np.append(o_tot,o)
        
        # Termination
        state_partial = stateSpace[x[0],:].reshape(1,len(stateSpace[x[0],:]))
        state = np.concatenate((state_partial,[[psi]]),1)
        prob_b = NN_termination(np.append(state,[[o]], axis=1)).numpy()
        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
        for i in range(1,prob_b_rescaled.shape[1]):
            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
        b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
        b_tot = np.append(b_tot,b)
        if b == 1:
            b_bool = True
        else:
            b_bool = False
        
        o_prob_tilde = np.empty((1,option_space))
        if b_bool == True:
            o_prob_tilde = NN_options(state).numpy()
        else:
            o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
            o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
        prob_o = o_prob_tilde
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
        o_tot = np.append(o_tot,o)
        
        for k in range(0,max_epoch):
            state_partial = stateSpace[x[k],:].reshape(1,len(stateSpace[x[k],:]))
            state = np.concatenate((state_partial,[[psi]]),1)
            # draw action
            prob_u = NN_actions(np.append(state,[[o]], axis=1)).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            
            # given action, draw next state
            x_k_possible=np.where(P[x[k],:,int(u)]!=0)
            prob = P[x[k],x_k_possible[0][:],int(u)]
            prob_rescaled = np.divide(prob,np.amin(prob))
            
            for i in range(1,prob_rescaled.shape[0]):
                prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
            draw=np.divide(np.random.rand(),np.amin(prob))
            index_x_plus1=np.amin(np.where(draw<prob_rescaled))
            x = np.append(x, x_k_possible[0][index_x_plus1])
            u_tot = np.append(u_tot,u)
            
            if x[k] == terminal_state1 or x[k] == terminal_state2:
                r = r + 1 
            
            # Randomly update the reward
            psi = UpdateReward(psi, x[k], terminal_state1, terminal_state2)
            psi_tot = np.append(psi_tot, psi)
            
            # Select Termination
            # Termination
            state_plus1_partial = stateSpace[x[k+1],:].reshape(1,len(stateSpace[x[k+1],:]))
            state_plus1 = np.concatenate((state_plus1_partial,[[psi]]),1)
            prob_b = NN_termination(np.append(state_plus1,[[o]], axis=1)).numpy()
            prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
            for i in range(1,prob_b_rescaled.shape[1]):
                prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
            draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
            b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
            b_tot = np.append(b_tot,b)
            if b == 1:
                b_bool = True
            else:
                b_bool = False
        
            o_prob_tilde = np.empty((1,option_space))
            if b_bool == True:
                o_prob_tilde = NN_options(state_plus1).numpy()
            else:
                o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
                o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
            prob_o = o_prob_tilde
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[1]):
                prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
            o_tot = np.append(o_tot,o)
            
        
        traj[t] = x
        control[t]=u_tot
        Option[t]=o_tot
        Termination[t]=b_tot
        psi_evolution[t] = psi_tot                
        reward = np.append(reward,r)

        
    return traj, control, Option, Termination, psi_evolution, reward               
            
            