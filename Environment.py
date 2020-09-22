#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:18:14 2020

@author: vittorio
"""

import StateSpace as ss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# Generate simplified world function
# =============================================================================

def Generate_world_subgoals_simplified():
    mapsize = [10, 11]
    map = np.zeros( (mapsize[0],mapsize[1]) )
    #define obstacles
    map[0,5] = 1
    map[3:7,5]=1;
    map[mapsize[0]-1,5]=1;

    #count trees
    ntrees=0;
    trees = np.empty((0,2),int)
    for i in range(0,mapsize[0]):
        for j in range(0,mapsize[1]):
            if map[i,j]==1:
                trees = np.append(trees, [[j, i]], 0)
                ntrees += 1

    #shooters
    # nshooters=1;
    # shooters = np.array([3, 2])
    # map[shooters[1],shooters[0]]=2

    #R1
    pick_up = np.array([1, 8])
    map[pick_up[1],pick_up[0]]=3

    #R2
    drop_off = np.array([9, 8])
    map[drop_off[1],drop_off[0]]=4

    #base
    base = np.array([1, 1])
    map[base[1],base[0]]=5

    plt.figure()
    plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
             [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
    plt.plot([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
             [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'k-')
    plt.plot([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
             [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'k-')
    # plt.plot([shooters[0], shooters[0], shooters[0]+1, shooters[0]+1, shooters[0]],
    #          [shooters[1], shooters[1]+1, shooters[1]+1, shooters[1], shooters[1]],'k-')

    for i in range(0,ntrees):
        plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

    plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
             [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
    plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
             [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
    plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
             [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'y')
    # plt.fill([shooters[0], shooters[0], shooters[0]+1, shooters[0]+1, shooters[0]],
    #          [shooters[1], shooters[1]+1, shooters[1]+1, shooters[1], shooters[1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'R1')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'R2')
    # plt.text(shooters[0]+0.5, shooters[1]+0.5, 'S')


    return map

def PlotOptimalSolution(map,stateSpace,u,name):

    mapsize = map.shape
    #count trees
    ntrees=0;
    trees = np.empty((0,2),int)
    shooters = np.empty((0,2),int)
    nshooters=0
    for i in range(0,mapsize[0]):
        for j in range(0,mapsize[1]):
            if map[i,j]==ss.TREE:
                trees = np.append(trees, [[j, i]], 0)
                ntrees += 1
            if map[i,j]==ss.SHOOTER:
                shooters = np.append(shooters, [[j, i]], 0)
                nshooters+=1

    #pickup station
    PickUpIndex=ss.R1StateIndex(stateSpace,map)
    i_pickup = stateSpace[PickUpIndex,0]
    j_pickup = stateSpace[PickUpIndex,1]
    pick_up = np.array([j_pickup, i_pickup])
    #base
    BaseIndex=ss.BaseStateIndex(stateSpace,map)
    i_base = stateSpace[BaseIndex,0]
    j_base = stateSpace[BaseIndex,1]
    base = np.array([j_base, i_base])
    #drop_off
    DropOffIndex = ss.R2StateIndex(stateSpace,map)
    i_dropoff = stateSpace[DropOffIndex,0]
    j_dropoff = stateSpace[DropOffIndex,1]
    drop_off = np.array([j_dropoff, i_dropoff])

    # PICK_UP
    plt.figure()
    plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                 [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
    plt.plot([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                 [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'k-')
    plt.plot([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'k-')

    for i in range(0,nshooters):
        plt.plot([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'k-')

    for i in range(0,ntrees):
        plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                 [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

    plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                 [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
    plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                 [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
    plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'y')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'R1')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'R2')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    for s in range(0,u.shape[0]):
        if u[s] == ss.NORTH:
            txt = u'\u2191'
        elif u[s] == ss.SOUTH:
            txt = u'\u2193'
        elif u[s] == ss.EAST:
            txt = u'\u2192'
        elif u[s] == ss.WEST:
            txt = u'\u2190'
        elif u[s] == ss.HOVER:
            txt = u'\u2715'
        plt.text(stateSpace[s,1]+0.3, stateSpace[s,0]+0.5,txt)
            
    plt.savefig(name, format='eps')



def VideoSimulation(map,stateSpace,u,states,psi,name_video):
    mapsize = map.shape
    #count trees
    ntrees=0;
    trees = np.empty((0,2),int)
    shooters = np.empty((0,2),int)
    nshooters=0
    for i in range(0,mapsize[0]):
        for j in range(0,mapsize[1]):
            if map[i,j]==ss.TREE:
                trees = np.append(trees, [[j, i]], 0)
                ntrees += 1
            if map[i,j]==ss.SHOOTER:
                shooters = np.append(shooters, [[j, i]], 0)
                nshooters+=1

    #pickup station
    PickUpIndex=ss.R1StateIndex(stateSpace,map)
    i_pickup = stateSpace[PickUpIndex,0]
    j_pickup = stateSpace[PickUpIndex,1]
    pick_up = np.array([j_pickup, i_pickup])
    #base
    BaseIndex=ss.BaseStateIndex(stateSpace,map)
    i_base = stateSpace[BaseIndex,0]
    j_base = stateSpace[BaseIndex,1]
    base = np.array([j_base, i_base])
    #drop_off
    DropOffIndex = ss.R2StateIndex(stateSpace,map)
    i_dropoff = stateSpace[DropOffIndex,0]
    j_dropoff = stateSpace[DropOffIndex,1]
    drop_off = np.array([j_dropoff, i_dropoff])

    fig = plt.figure()
    plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                 [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
    plt.plot([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                 [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'k-')
    plt.plot([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'k-')

    for i in range(0,nshooters):
        plt.plot([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'k-')

    for i in range(0,ntrees):
        plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                 [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

    plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                 [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
    plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                 [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
    plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'y')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'R1')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'R2')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    ims = []
    for s in range(0,len(u)):
        if psi[s]==0:
            im2, = plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                           [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'m')
            im3, = plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                           [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'y')
        if psi[s]==1:
            im2, = plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                           [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
            im3, = plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                           [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'m')
        if psi[s]==2:
            im2, = plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                           [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'m')
            im3, = plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                           [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'m')
        if psi[s]==3:
            im2, = plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                           [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
            im3, = plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                           [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'y')
             
        if u[s] == ss.NORTH:
            txt = u'\u2191'
        elif u[s] == ss.SOUTH:
            txt = u'\u2193'
        elif u[s] == ss.EAST:
            txt = u'\u2192'
        elif u[s] == ss.WEST:
            txt = u'\u2190'
        elif u[s] == ss.HOVER:
            txt = u'\u2715'
        im1 = plt.text(stateSpace[states[s],1]+0.3, stateSpace[states[s],0]+0.1, txt, fontsize=25)
        ims.append([im1,im2,im3])
        
    ani = animation.ArtistAnimation(fig, ims, interval=1200, blit=True,
                                    repeat_delay=2000)
    ani.save(name_video)
    
    
def HILVideoSimulation(map,stateSpace,u,states,o,psi,name_video):
    mapsize = map.shape
    #count trees
    ntrees=0;
    trees = np.empty((0,2),int)
    shooters = np.empty((0,2),int)
    nshooters=0
    for i in range(0,mapsize[0]):
        for j in range(0,mapsize[1]):
            if map[i,j]==ss.TREE:
                trees = np.append(trees, [[j, i]], 0)
                ntrees += 1
            if map[i,j]==ss.SHOOTER:
                shooters = np.append(shooters, [[j, i]], 0)
                nshooters+=1

    #pickup station
    PickUpIndex=ss.R1StateIndex(stateSpace,map)
    i_pickup = stateSpace[PickUpIndex,0]
    j_pickup = stateSpace[PickUpIndex,1]
    pick_up = np.array([j_pickup, i_pickup])
    #base
    BaseIndex=ss.BaseStateIndex(stateSpace,map)
    i_base = stateSpace[BaseIndex,0]
    j_base = stateSpace[BaseIndex,1]
    base = np.array([j_base, i_base])
    #drop_off
    DropOffIndex = ss.R2StateIndex(stateSpace,map)
    i_dropoff = stateSpace[DropOffIndex,0]
    j_dropoff = stateSpace[DropOffIndex,1]
    drop_off = np.array([j_dropoff, i_dropoff])

    fig = plt.figure(4)
    plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                 [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
    plt.plot([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                 [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'k-')
    plt.plot([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'k-')

    for i in range(0,nshooters):
        plt.plot([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'k-')

    for i in range(0,ntrees):
        plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                 [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

    plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                 [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
    plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                 [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
    plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'y')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'R1')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'R2')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    ims = []
    for s in range(0,len(u)):
        if psi[s]==0:
            im2, = plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                           [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'m')
            im3, = plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                           [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'y')
        if psi[s]==1:
            im2, = plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                           [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
            im3, = plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                           [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'m')
        if psi[s]==2:
            im2, = plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                           [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'m')
            im3, = plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                           [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'m')
        if psi[s]==3:
            im2, = plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                           [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
            im3, = plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                           [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'y')
        if u[s] == ss.NORTH:
            txt = u'\u2191'
        elif u[s] == ss.SOUTH:
            txt = u'\u2193'
        elif u[s] == ss.EAST:
            txt = u'\u2192'
        elif u[s] == ss.WEST:
            txt = u'\u2190'
        elif u[s] == ss.HOVER:
            txt = u'\u2715'
        if o[s]==0:
            c = 'c'
        elif o[s]==1:
            c = 'm'
        elif o[s]==2:
            c = 'y'         
        im1 = plt.text(stateSpace[states[s],1]+0.3, stateSpace[states[s],0]+0.1, txt, fontsize=20, backgroundcolor=c)
        ims.append([im1,im2,im3])
        
    ani = animation.ArtistAnimation(fig, ims, interval=1200, blit=True,
                                repeat_delay=2000)
    ani.save(name_video)
    
    
  
def PlotOptimalOptions(map,stateSpace, o, name):

    mapsize = map.shape
    #count trees
    ntrees=0;
    trees = np.empty((0,2),int)
    shooters = np.empty((0,2),int)
    nshooters=0
    for i in range(0,mapsize[0]):
        for j in range(0,mapsize[1]):
            if map[i,j]==ss.TREE:
                trees = np.append(trees, [[j, i]], 0)
                ntrees += 1
            if map[i,j]==ss.SHOOTER:
                shooters = np.append(shooters, [[j, i]], 0)
                nshooters+=1

    #pickup station
    PickUpIndex=ss.R1StateIndex(stateSpace,map)
    i_pickup = stateSpace[PickUpIndex,0]
    j_pickup = stateSpace[PickUpIndex,1]
    pick_up = np.array([j_pickup, i_pickup])
    #base
    BaseIndex=ss.BaseStateIndex(stateSpace,map)
    i_base = stateSpace[BaseIndex,0]
    j_base = stateSpace[BaseIndex,1]
    base = np.array([j_base, i_base])
    #drop_off
    DropOffIndex = ss.R2StateIndex(stateSpace,map)
    i_dropoff = stateSpace[DropOffIndex,0]
    j_dropoff = stateSpace[DropOffIndex,1]
    drop_off = np.array([j_dropoff, i_dropoff])

    plt.figure()
    plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                 [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
    plt.plot([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                 [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'k-')
    plt.plot([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'k-')

    for i in range(0,nshooters):
        plt.plot([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'k-')

    for i in range(0,ntrees):
        plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                 [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

    plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                 [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
    plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
                 [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
    plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'y')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'R1')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'R2')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    for s in range(0,len(o)):
        if o[s]==0:
            c = 'c'
        elif o[s]==1:
            c = 'lime'
        elif o[s]==2:
            c = 'y'    
        plt.fill([stateSpace[s,1], stateSpace[s,1], stateSpace[s,1]+0.9, stateSpace[s,1]+0.9, stateSpace[s,1]],
                 [stateSpace[s,0], stateSpace[s,0]+0.9, stateSpace[s,0]+0.9, stateSpace[s,0], stateSpace[s,0]],c)
 
            
    plt.savefig(name, format='eps')

     

    