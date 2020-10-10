#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:39:39 2020

@author: vittorio
"""
import tensorflow as tf 
import numpy as np
import argparse
import os
import Simulation as sim
from tensorflow import keras
import tensorflow.keras.backend as kb
import BehavioralCloning as bc
import concurrent.futures
import StateSpace as ss

def NN_options(option_space,size_input):
    model = keras.Sequential([
    keras.layers.Dense(100, activation='relu', input_shape=(size_input,),
                      kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                      bias_initializer=keras.initializers.Zeros()),
    keras.layers.Dense(option_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresHIL/model_NN_options.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

def NN_actions(action_space, size_input):
    model = keras.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=(size_input,),
                       kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                      bias_initializer=keras.initializers.Zeros()),
    keras.layers.Dense(action_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresHIL/model_NN_actions.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

def NN_termination(termination_space, size_input):
    model = keras.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=(size_input,),
                      kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                      bias_initializer=keras.initializers.Zeros()),
    keras.layers.Dense(termination_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresHIL/model_NN_termination.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

class Environment_specs:
    def __init__(self, P, stateSpace, map):
        self.P = P
        self.stateSpace = stateSpace
        self.map = map

class DDO:
    def __init__(self, Labels, TrainingSet, size_input, action_space, option_space, termination_space, N1, N2, zeta, mu,
                 NN_options, NN_actions, NN_termination, lambda_gain, M_step_epoch, size_batch, optimizer):
        self.labels = Labels
        self.TrainingSet = TrainingSet
        self.size_input = size_input
        self.action_space = action_space
        self.option_space = option_space
        self.termination_space = termination_space
        self.N1 = N1
        self.N2 = N2
        self.zeta = zeta
        self.mu = mu
        self.NN_options = NN_options
        self.NN_actions = NN_actions
        self.NN_termination = NN_termination
        self.lambda_gain = lambda_gain
        self.epochs = M_step_epoch
        self.size_batch = size_batch
        self.optimizer = optimizer
        
    def Pi_hi(ot, Pi_hi_parameterization, state):
        Pi_hi = Pi_hi_parameterization(state)
        o_prob = Pi_hi[0,ot]
        return o_prob

    def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space):
        if b == True:
            o_prob_tilde = DDO.Pi_hi(ot, Pi_hi_parameterization, state)
        elif ot == ot_past:
            o_prob_tilde = 1-zeta+np.divide(zeta,option_space)
        else:
            o_prob_tilde = np.divide(zeta,option_space)
        
        return o_prob_tilde

    def Pi_lo(a, Pi_lo_parameterization, state):
        Pi_lo = Pi_lo_parameterization(state)
        a_prob = Pi_lo[0,int(a)]
    
        return a_prob

    def Pi_b(b, Pi_b_parameterization, state):
        Pi_b = Pi_b_parameterization(state)
        if b == True:
            b_prob = Pi_b[0,1]
        else:
            b_prob = Pi_b[0,0]
        return b_prob
    
    def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, zeta, option_space):
        Pi_hi_eval = DDO.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space)
        Pi_lo_eval = DDO.Pi_lo(a, Pi_lo_parameterization, state)
        Pi_b_eval = DDO.Pi_b(b, Pi_b_parameterization, state)
        output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
        return output
    
    def ForwardRecursion(alpha_past, a, Pi_hi_parameterization, Pi_lo_parameterization,
                         Pi_b_parameterization, state, zeta, option_space, termination_space):
        # =============================================================================
        #     alpha is the forward message: alpha.shape()= [option_space, termination_space]
        # =============================================================================
        alpha = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot = i1
            for i2 in range(termination_space):
                if i2 == 1:
                    bt=True
                else:
                    bt=False
            
                Pi_comb = np.zeros(option_space)
                for ot_past in range(option_space):
                    Pi_comb[ot_past] = DDO.Pi_combined(ot, ot_past, a, bt, 
                                                       Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                       state, zeta, option_space)
                alpha[ot,i2] = np.dot(alpha_past[:,0],Pi_comb)+np.dot(alpha_past[:,1],Pi_comb)  
        alpha = np.divide(alpha,np.sum(alpha))
            
        return alpha
    
    def ForwardFirstRecursion(mu, a, Pi_hi_parameterization, Pi_lo_parameterization,
                              Pi_b_parameterization, state, zeta, option_space, termination_space):
        # =============================================================================
        #     alpha is the forward message: alpha.shape()=[option_space, termination_space]
        #   mu is the initial distribution over options: mu.shape()=[1,option_space]
        # =============================================================================
        alpha = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot = i1
            for i2 in range(termination_space):
                if i2 == 1:
                    bt=True
                else:
                    bt=False
            
                Pi_comb = np.zeros(option_space)
                for ot_past in range(option_space):
                    Pi_comb[ot_past] = DDO.Pi_combined(ot, ot_past, a, bt, 
                                                       Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                       state, zeta, option_space)
                    alpha[ot,i2] = np.dot(mu, Pi_comb[:])    
        alpha = np.divide(alpha, np.sum(alpha))
            
        return alpha

    def BackwardRecursion(beta_next, a, Pi_hi_parameterization, Pi_lo_parameterization,
                          Pi_b_parameterization, state, zeta, option_space, termination_space):
        # =============================================================================
        #     beta is the backward message: beta.shape()= [option_space, termination_space]
        # =============================================================================
        beta = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot = i1
            for i2 in range(termination_space):
                for i1_next in range(option_space):
                    ot_next = i1_next
                    for i2_next in range(termination_space):
                        if i2 == 1:
                            b_next=True
                        else:
                            b_next=False
                        beta[i1,i2] = beta[i1,i2] + beta_next[ot_next,i2_next]*DDO.Pi_combined(ot_next, ot, a, b_next, 
                                                                                               Pi_hi_parameterization, Pi_lo_parameterization[ot_next], 
                                                                                               Pi_b_parameterization[ot], state, zeta, option_space)
        beta = np.divide(beta,np.sum(beta))
    
        return beta

    def Alpha(self):
        alpha = np.empty((self.option_space,self.termination_space,len(self.TrainingSet)))
        for t in range(len(self.TrainingSet)):
            print('alpha iter', t+1, '/', len(self.TrainingSet))
            if t ==0:
                state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
                action = self.labels[t]
                alpha[:,:,t] = DDO.ForwardFirstRecursion(self.mu, action, self.NN_options, 
                                                         self.NN_actions, self.NN_termination, 
                                                         state, self.zeta, self.option_space, self.termination_space)
            else:
                state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
                action = self.labels[t]
                alpha[:,:,t] = DDO.ForwardRecursion(alpha[:,:,t-1], action, self.NN_options, 
                                                    self.NN_actions, self.NN_termination, 
                                                    state, self.zeta, self.option_space, self.termination_space)
           
        return alpha

    def Beta(self):
        beta = np.empty((self.option_space,self.termination_space,len(self.TrainingSet)))
        beta[:,:,len(self.TrainingSet)-1] = np.divide(np.ones((self.option_space,self.termination_space)),2*self.option_space)
    
        for t_raw in range(len(self.TrainingSet)-1):
            t = len(self.TrainingSet) - (t_raw+1)
            print('beta iter', t_raw+1, '/', len(self.TrainingSet)-1)
            state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
            action = self.labels[t]
            beta[:,:,t-1] = DDO.BackwardRecursion(beta[:,:,t], action, self.NN_options, 
                                                  self.NN_actions, self.NN_termination, state, self.zeta, 
                                                  self.option_space, self.termination_space)
        
        return beta

    def Smoothing(option_space, termination_space, alpha, beta):
        gamma = np.empty((option_space, termination_space))
        for i1 in range(option_space):
            ot=i1
            for i2 in range(termination_space):
                gamma[ot,i2] = alpha[ot,i2]*beta[ot,i2]     
            gamma = np.divide(gamma,np.sum(gamma))
    
        return gamma

    def DoubleSmoothing(beta, alpha, a, Pi_hi_parameterization, Pi_lo_parameterization, 
                    Pi_b_parameterization, state, zeta, option_space, termination_space):
        gamma_tilde = np.empty((option_space, termination_space))
        for i1_past in range(option_space):
            ot_past = i1_past
            for i2 in range(termination_space):
                if i2 == 1:
                    b=True
                else:
                    b=False
                for i1 in range(option_space):
                    ot = i1
                    gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2] + beta[ot,i2]*DDO.Pi_combined(ot, ot_past, a, b, 
                                                                                                Pi_hi_parameterization, Pi_lo_parameterization[ot], 
                                                                                                Pi_b_parameterization[ot_past], state, zeta, option_space)
                gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2]*np.sum(alpha[ot_past,:])
        gamma_tilde = np.divide(gamma_tilde,np.sum(gamma_tilde))
    
        return gamma_tilde

    def Gamma(self, alpha, beta):
        gamma = np.empty((self.option_space,self.termination_space,len(self.TrainingSet)))
        for t in range(len(self.TrainingSet)):
            print('gamma iter', t+1, '/', len(self.TrainingSet))
            gamma[:,:,t]=DDO.Smoothing(self.option_space, self.termination_space, alpha[:,:,t], beta[:,:,t])
        
        return gamma

    def GammaTilde(self, beta, alpha):
        gamma_tilde = np.zeros((self.option_space,self.termination_space,len(self.TrainingSet)))
        for t in range(1,len(self.TrainingSet)):
            print('gamma tilde iter', t, '/', len(self.TrainingSet)-1)
            state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
            action = self.labels[t]
            gamma_tilde[:,:,t]=DDO.DoubleSmoothing(beta[:,:,t], alpha[:,:,t-1], action, 
                                                   self.NN_options, self.NN_actions, self.NN_termination, 
                                                   state, self.zeta, self.option_space, self.termination_space)
        return gamma_tilde
        
    def GammaTildeReshape(gamma_tilde, option_space):
        T = gamma_tilde.shape[2]
        gamma_tilde_reshaped_array = np.empty((T-1,2,option_space))
        for i in range(option_space):
            gamma_tilde_reshaped = gamma_tilde[i,:,1:]
            gamma_tilde_reshaped_array[:,:,i] = gamma_tilde_reshaped.reshape(T-1,2)
            
        return gamma_tilde_reshaped_array
    
    def GammaReshapeActions(T, option_space, action_space, gamma, labels):
        gamma_actions_array = np.empty((T, action_space, option_space))
        for k in range(option_space):
            gamma_reshaped_option = gamma[k,:,:]    
            gamma_reshaped_option = np.sum(gamma_reshaped_option,0)
            gamma_actions = np.empty((int(T),action_space))
            for i in range(T):
                for j in range(action_space):
                    if int(labels[i])==j:
                        gamma_actions[i,j]=gamma_reshaped_option[i]
                    else:
                        gamma_actions[i,j] = 0
            gamma_actions_array[:,:,k] = gamma_actions
            
        return gamma_actions_array
    
    def GammaReshapeOptions(gamma):
        gamma_reshaped_options = gamma[:,1,:]
        gamma_reshaped_options = np.transpose(gamma_reshaped_options)
        
        return gamma_reshaped_options
    
    def Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, 
                    NN_termination, NN_options, NN_actions, T, TrainingSet):
        loss = 0
        option_space = len(NN_actions)
        for i in range(option_space):
            pi_b = NN_termination[i](TrainingSet[:],training=True)
            loss = loss -kb.sum(gamma_tilde_reshaped[:,:,i]*kb.log(pi_b[:]))/(T)
            pi_lo = NN_actions[i](TrainingSet,training=True)
            loss = loss -(kb.sum(gamma_actions[:,:,i]*kb.log(pi_lo)))/(T)
            
        pi_hi = NN_options(TrainingSet,training=True)
        loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
        loss = loss + loss_options
    
        return loss    

    
    def OptimizeLoss(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions):
        option_space = len(NN_actions)
        weights = []
        loss = 0
        
        T = self.TrainingSet.shape[0]
        
        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
        
            with tf.GradientTape() as tape:
                for i in range(option_space):
                    weights.append(self.NN_termination[i].trainable_weights)
                    weights.append(self.NN_actions[i].trainable_weights)
                weights.append(self.NN_options.trainable_weights)
                tape.watch(weights)
                loss = DDO.Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, 
                                self.NN_termination, self.NN_options, self.NN_actions, T, self.TrainingSet)
            
            grads = tape.gradient(loss,weights)
            j=0
            for i in range(0,2*option_space,2):
                self.optimizer.apply_gradients(zip(grads[i][:], self.NN_termination[j].trainable_weights))
                self.optimizer.apply_gradients(zip(grads[i+1][:], self.NN_actions[j].trainable_weights))
                j = j+1
            self.optimizer.apply_gradients(zip(grads[-1][:], self.NN_options.trainable_weights))
            print('options loss:', float(loss))
        
        return loss        
    

    def OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions):
        option_space = len(self.NN_actions)
        weights = []
        loss = 0
        
        n_batches = np.int(self.TrainingSet.shape[0]/self.size_batch)

        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            
            for n in range(n_batches):
                print("\n Batch %d" % (n+1,))
        
                with tf.GradientTape() as tape:
                    for i in range(option_space):
                        weights.append(self.NN_termination[i].trainable_weights)
                        weights.append(self.NN_actions[i].trainable_weights)
                    weights.append(self.NN_options.trainable_weights)
                    tape.watch(weights)
                    loss = DDO.Loss(gamma_tilde_reshaped[n*self.size_batch:(n+1)*self.size_batch,:,:], 
                                    gamma_reshaped_options[n*self.size_batch:(n+1)*self.size_batch,:], 
                                    gamma_actions[n*self.size_batch:(n+1)*self.size_batch,:,:], 
                                    self.NN_termination, self.NN_options, self.NN_actions, self.size_batch, 
                                    self.TrainingSet[n*self.size_batch:(n+1)*self.size_batch,:])
            
                grads = tape.gradient(loss,weights)
                j=0
                for i in range(0,2*option_space,2):
                    self.optimizer.apply_gradients(zip(grads[i][:], self.NN_termination[j].trainable_weights))
                    self.optimizer.apply_gradients(zip(grads[i+1][:], self.NN_actions[j].trainable_weights))
                    j = j+1
                self.optimizer.apply_gradients(zip(grads[-1][:], self.NN_options.trainable_weights))
                print('loss:', float(loss))
        
        return loss      
    
    def RegularizedLoss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, 
                    NN_termination, NN_options, NN_actions, T, TrainingSet, lambda_gain):
        loss = 0
        option_space = len(NN_actions)
        
        # Regularization 
        Reg = lambda_gain*kb.sum(-kb.sum(NN_options(TrainingSet,training=True)*kb.log(NN_options(TrainingSet,training=True)),1)/T,0)
        
        for i in range(option_space):
            pi_b = NN_termination[i](TrainingSet[:],training=True)
            loss = loss -kb.sum(gamma_tilde_reshaped[:,:,i]*kb.log(pi_b[:]))/(T)
            pi_lo = NN_actions[i](TrainingSet,training=True)
            loss = loss -(kb.sum(gamma_actions[:,:,i]*kb.log(pi_lo)))/(T)
            
        pi_hi = NN_options(TrainingSet,training=True)
        loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
        loss = loss + loss_options - Reg
    
        return loss    

    
    def OptimizeRegularizedLoss(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions):
        option_space = len(NN_actions)
        weights = []
        loss = 0
        
        T = self.TrainingSet.shape[0]
        
        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
        
            with tf.GradientTape() as tape:
                for i in range(option_space):
                    weights.append(self.NN_termination[i].trainable_weights)
                    weights.append(self.NN_actions[i].trainable_weights)
                weights.append(self.NN_options.trainable_weights)
                tape.watch(weights)
                loss = DDO.RegularizedLoss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, 
                                           self.NN_termination, self.NN_options, self.NN_actions, T, self.TrainingSet, self.lambda_gain)
            
            grads = tape.gradient(loss,weights)
            j=0
            for i in range(0,2*option_space,2):
                self.optimizer.apply_gradients(zip(grads[i][:], self.NN_termination[j].trainable_weights))
                self.optimizer.apply_gradients(zip(grads[i+1][:], self.NN_actions[j].trainable_weights))
                j = j+1
            self.optimizer.apply_gradients(zip(grads[-1][:], self.NN_options.trainable_weights))
            print('options loss:', float(loss))
        
        return loss        
    
    def OptimizeRegularizedLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions):
        option_space = len(self.NN_actions)
        weights = []
        loss = 0
        
        n_batches = np.int(self.TrainingSet.shape[0]/self.size_batch)

        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            
            for n in range(n_batches):
                print("\n Batch %d" % (n+1,))
        
                with tf.GradientTape() as tape:
                    for i in range(option_space):
                        weights.append(self.NN_termination[i].trainable_weights)
                        weights.append(self.NN_actions[i].trainable_weights)
                    weights.append(self.NN_options.trainable_weights)
                    tape.watch(weights)
                    loss = DDO.RegularizedLoss(gamma_tilde_reshaped[n*self.size_batch:(n+1)*self.size_batch,:,:], 
                                               gamma_reshaped_options[n*self.size_batch:(n+1)*self.size_batch,:], 
                                               gamma_actions[n*self.size_batch:(n+1)*self.size_batch,:,:], 
                                               self.NN_termination, self.NN_options, self.NN_actions, self.size_batch, 
                                               self.TrainingSet[n*self.size_batch:(n+1)*self.size_batch,:], self.lambda_gain)
            
                grads = tape.gradient(loss,weights)
                j=0
                for i in range(0,2*option_space,2):
                    self.optimizer.apply_gradients(zip(grads[i][:], self.NN_termination[j].trainable_weights))
                    self.optimizer.apply_gradients(zip(grads[i+1][:], self.NN_actions[j].trainable_weights))
                    j = j+1
                self.optimizer.apply_gradients(zip(grads[-1][:], self.NN_options.trainable_weights))
                print('loss:', float(loss))
        
        return loss    
    
    def BaumWelch(self):
        
        T = self.TrainingSet.shape[0]

        for n in range(self.N1):
            print('iter Regularized loss', n+1, '/', self.N1)
        
            alpha = DDO.Alpha(self)
            beta = DDO.Beta(self)
            gamma = DDO.Gamma(self, alpha, beta)
            gamma_tilde = DDO.GammaTilde(self, beta, alpha)
        
            print('Expectation done')
            print('Starting maximization step')
            
            gamma_tilde_reshaped = DDO.GammaTildeReshape(gamma_tilde, self.option_space)
            gamma_actions = DDO.GammaReshapeActions(T, self.option_space, self.action_space, gamma, self.labels)
            gamma_reshaped_options = DDO.GammaReshapeOptions(gamma)
    

            loss = DDO.OptimizeRegularizedLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions)
            
        for n in range(self.N2):
            print('iter Loss', n+1, '/', self.N2)
        
            alpha = DDO.Alpha(self)
            beta = DDO.Beta(self)
            gamma = DDO.Gamma(self, alpha, beta)
            gamma_tilde = DDO.GammaTilde(self, beta, alpha)
        
            print('Expectation done')
            print('Starting maximization step')
            
            gamma_tilde_reshaped = DDO.GammaTildeReshape(gamma_tilde, self.option_space)
            gamma_actions = DDO.GammaReshapeActions(T, self.option_space, self.action_space, gamma, self.labels)
            gamma_reshaped_options = DDO.GammaReshapeOptions(gamma)
    

            loss = DDO.OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions)

        print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

        
        return self.NN_termination, self.NN_actions, self.NN_options
    
    class Simulation:
        
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
                prob_b = NN_termination[o](state).numpy()
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
                    prob_u = NN_actions[o](state).numpy()
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
                    psi = DDO.Simulation.UpdateReward(psi, x[k], terminal_state1, terminal_state2)
                    psi_tot = np.append(psi_tot, psi)
            
                    # Select Termination
                    # Termination
                    state_plus1_partial = stateSpace[x[k+1],:].reshape(1,len(stateSpace[x[k+1],:]))
                    state_plus1 = np.concatenate((state_plus1_partial,[[psi]]),1)
                    prob_b = NN_termination[o](state_plus1).numpy()
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