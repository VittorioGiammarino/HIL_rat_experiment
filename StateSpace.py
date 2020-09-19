#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:22:26 2020

@author: vittorio
"""

import numpy as np


# Global problem parameters
# Do not add or remove any global parameter 
GAMMA  = 0.2 # Shooter gamma factor
R = 2 # Shooter range
Nc = 10 # Time steps required to bring drone to base when it crashes
P_WIND = 0.1 # Gust of wind probability

# IDs of elements in the map matrix
FREE = 0
TREE = 1
SHOOTER = 2
PICK_UP = 3
DROP_OFF = 4
BASE = 5

# Index of each action in the P and G matrices. Use this ordering
NORTH  = 0
SOUTH = 1
EAST = 2
WEST = 3
HOVER = 4

def GenerateStateSpace(map):
    print('Generate State Space')

    stateSpace = np.empty((0,2),int)

    for m in range(0,map.shape[0]):
        for n in range(0,map.shape[1]):
            if map[m,n] != TREE:
                stateSpace = np.append(stateSpace, [[m, n]], 0)
            
    return stateSpace


def BaseStateIndex(stateSpace, map):
    
    global BASE
    
    K = stateSpace.shape[0];
    
    for i in range(0,map.shape[0]):
        for j in range(0,map.shape[1]):
            if map[i,j]==BASE:
                m=i
                n=j
                break
            
    for i in range(0,K):
        if stateSpace[i,0]==m and stateSpace[i,1]==n:
            stateIndex = i
            break
    
    return stateIndex

def R2StateIndex(stateSpace, map):
    
    global DROP_OFF
    
    K = stateSpace.shape[0];
    
    for i in range(0,map.shape[0]):
        for j in range(0,map.shape[1]):
            if map[i,j]==DROP_OFF:
                m=i
                n=j
                break
            
    for i in range(0,K):
        if stateSpace[i,0]==m and stateSpace[i,1]==n:
            stateIndex = i
            break
    
    return stateIndex

def R1StateIndex(stateSpace, map):
    
    global PICK_UP
    
    K = stateSpace.shape[0];
    
    for i in range(0,map.shape[0]):
        for j in range(0,map.shape[1]):
            if map[i,j]==PICK_UP:
                m=i
                n=j
                break
            
    for i in range(0,K):
        if stateSpace[i,0]==m and stateSpace[i,1]==n:
            stateIndex = i
            break
    
    return stateIndex


def FindStateIndex(stateSpace, value):
    
    K = stateSpace.shape[0];
    stateIndex = 0
    
    for k in range(0,K):
        if stateSpace[k,0]==value[0] and stateSpace[k,1]==value[1]:
            stateIndex = k
    
    return stateIndex

