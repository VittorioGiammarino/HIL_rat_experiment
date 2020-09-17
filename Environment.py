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
    map[0:4,5]=1;
    map[6:mapsize[0],5]=1;

    #count trees
    ntrees=0;
    trees = np.empty((0,2),int)
    for i in range(0,mapsize[0]):
        for j in range(0,mapsize[1]):
            if map[i,j]==1:
                trees = np.append(trees, [[j, i]], 0)
                ntrees += 1

    #shooters
    nshooters=1;
    shooters = np.array([3, 2])
    map[shooters[1],shooters[0]]=2

    #pick up
    pick_up = np.array([7, 1])
    map[pick_up[1],pick_up[0]]=3

    #drop off
    drop_off = np.array([1, 8])
    map[drop_off[1],drop_off[0]]=4

    #base
    base = np.array([mapsize[1]-2, mapsize[0]-2])
    map[base[1],base[0]]=5

    plt.figure()
    plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
             [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
    plt.plot([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
             [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'k-')
    plt.plot([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
             [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'k-')
    plt.plot([shooters[0], shooters[0], shooters[0]+1, shooters[0]+1, shooters[0]],
             [shooters[1], shooters[1]+1, shooters[1]+1, shooters[1], shooters[1]],'k-')

    for i in range(0,ntrees):
        plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

    plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
             [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
    plt.fill([pick_up[0], pick_up[0], pick_up[0]+1, pick_up[0]+1, pick_up[0]],
             [pick_up[1], pick_up[1]+1, pick_up[1]+1, pick_up[1], pick_up[1]],'y')
    plt.fill([drop_off[0], drop_off[0], drop_off[0]+1, drop_off[0]+1, drop_off[0]],
             [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'b')
    plt.fill([shooters[0], shooters[0], shooters[0]+1, shooters[0]+1, shooters[0]],
             [shooters[1], shooters[1]+1, shooters[1]+1, shooters[1], shooters[1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'P')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'D')
    plt.text(shooters[0]+0.5, shooters[1]+0.5, 'S')


    return map

def PlotOptimalSolution(map,stateSpace,u,name_pick_up, name_drop_off):

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
    PickUpIndex=ss.PickUpStateIndex(stateSpace,map)
    i_pickup = stateSpace[PickUpIndex,0]
    j_pickup = stateSpace[PickUpIndex,1]
    pick_up = np.array([j_pickup, i_pickup])
    #base
    BaseIndex=ss.BaseStateIndex(stateSpace,map)
    i_base = stateSpace[BaseIndex,0]
    j_base = stateSpace[BaseIndex,1]
    base = np.array([j_base, i_base])
    #drop_off
    DropOffIndex = ss.TerminalStateIndex(stateSpace,map)
    i_dropoff = stateSpace[DropOffIndex,0]
    j_dropoff = stateSpace[DropOffIndex,1]
    drop_off = np.array([j_dropoff, i_dropoff])
    # split the solution in pick up and drop off
    u_pick = np.zeros(int(u.shape[0]/2))
    u_drop = np.zeros(int(u.shape[0]/2))
    j=0
    for i in range(1,u.shape[0],2):
        u_pick[j]=u[i-1]
        u_drop[j]=u[i]
        j+=1
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
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'b')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'P')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'D')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    p=0
    for s in range(0,u_pick.shape[0]):
        if u_pick[s] == ss.NORTH:
            txt = u'\u2191'
        elif u_pick[s] == ss.SOUTH:
            txt = u'\u2193'
        elif u_pick[s] == ss.EAST:
            txt = u'\u2192'
        elif u_pick[s] == ss.WEST:
            txt = u'\u2190'
        elif u_pick[s] == ss.HOVER:
            txt = u'\u2715'
        plt.text(stateSpace[p,1]+0.3, stateSpace[p,0]+0.5,txt)
        if p < u.shape[0]-1:
            p=p+2
            
    plt.savefig(name_pick_up, format='eps')

    # DROP_OFF
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
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'b')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'P')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'D')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    p=1
    for s in range(0,u_drop.shape[0]):
        if u_drop[s] == ss.NORTH:
            txt = u'\u2191'
        elif u_drop[s] == ss.SOUTH:
            txt = u'\u2193'
        elif u_drop[s] == ss.EAST:
            txt = u'\u2192'
        elif u_drop[s] == ss.WEST:
            txt = u'\u2190'
        elif u_drop[s] == ss.HOVER:
            txt = u'\u2715'
        plt.text(stateSpace[p,1]+0.3, stateSpace[p,0]+0.5,txt)
        if p < u.shape[0]:
            p=p+2
            
    plt.savefig(name_drop_off, format='eps')

def VideoSimulation(map,stateSpace,u,states,name_video):
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
    PickUpIndex=ss.PickUpStateIndex(stateSpace,map)
    i_pickup = stateSpace[PickUpIndex,0]
    j_pickup = stateSpace[PickUpIndex,1]
    pick_up = np.array([j_pickup, i_pickup])
    #base
    BaseIndex=ss.BaseStateIndex(stateSpace,map)
    i_base = stateSpace[BaseIndex,0]
    j_base = stateSpace[BaseIndex,1]
    base = np.array([j_base, i_base])
    #drop_off
    DropOffIndex = ss.TerminalStateIndex(stateSpace,map)
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
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'b')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'P')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'D')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    ims = []
    for s in range(0,len(states)):
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
        im = plt.text(stateSpace[states[s],1]+0.3, stateSpace[states[s],0]+0.1, txt, fontsize=25)
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=1200, blit=True,
                                repeat_delay=2000)
    ani.save(name_video)
    
    plt.show()
    
def HILVideoSimulation(map,stateSpace,u,states,o,name_video):
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
    PickUpIndex=ss.PickUpStateIndex(stateSpace,map)
    i_pickup = stateSpace[PickUpIndex,0]
    j_pickup = stateSpace[PickUpIndex,1]
    pick_up = np.array([j_pickup, i_pickup])
    #base
    BaseIndex=ss.BaseStateIndex(stateSpace,map)
    i_base = stateSpace[BaseIndex,0]
    j_base = stateSpace[BaseIndex,1]
    base = np.array([j_base, i_base])
    #drop_off
    DropOffIndex = ss.TerminalStateIndex(stateSpace,map)
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
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'b')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'P')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'D')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    ims = []
    for s in range(0,len(states)):
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
        im = plt.text(stateSpace[states[s],1]+0.3, stateSpace[states[s],0]+0.1, txt, fontsize=20, backgroundcolor=c)
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=1200, blit=True,
                                repeat_delay=2000)
    ani.save(name_video)
    
    plt.show()
  
def PlotOptimalOptions(map,stateSpace,o,name_pick_up, name_drop_off):

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
    PickUpIndex=ss.PickUpStateIndex(stateSpace,map)
    i_pickup = stateSpace[PickUpIndex,0]
    j_pickup = stateSpace[PickUpIndex,1]
    pick_up = np.array([j_pickup, i_pickup])
    #base
    BaseIndex=ss.BaseStateIndex(stateSpace,map)
    i_base = stateSpace[BaseIndex,0]
    j_base = stateSpace[BaseIndex,1]
    base = np.array([j_base, i_base])
    #drop_off
    DropOffIndex = ss.TerminalStateIndex(stateSpace,map)
    i_dropoff = stateSpace[DropOffIndex,0]
    j_dropoff = stateSpace[DropOffIndex,1]
    drop_off = np.array([j_dropoff, i_dropoff])
    # split the solution in pick up and drop off
    o_pick = np.zeros(int(o.shape[0]/2))
    o_drop = np.zeros(int(o.shape[0]/2))
    j=0
    for i in range(1,o.shape[0],2):
        o_pick[j]=o[i-1]
        o_drop[j]=o[i]
        j+=1
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
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'b')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'P')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'D')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    p=0
    for s in range(0,o_pick.shape[0]):
        if o_pick[s]==0:
            c = 'c'
        elif o_pick[s]==1:
            c = 'm'
        elif o_pick[s]==2:
            c = 'y'    
        plt.fill([stateSpace[p,1], stateSpace[p,1], stateSpace[p,1]+0.9, stateSpace[p,1]+0.9, stateSpace[p,1]],
                 [stateSpace[p,0], stateSpace[p,0]+0.9, stateSpace[p,0]+0.9, stateSpace[p,0], stateSpace[p,0]],c)
        if p < o.shape[0]-1:
            p=p+2
            
    plt.savefig(name_pick_up, format='eps')

    # DROP_OFF
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
                 [drop_off[1], drop_off[1]+1, drop_off[1]+1, drop_off[1], drop_off[1]],'b')

    for i in range(0,nshooters):
        plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                 [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

    for i in range(0,ntrees):
        plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'g')

    plt.text(base[0]+0.5, base[1]+0.5, 'B')
    plt.text(pick_up[0]+0.5, pick_up[1]+0.5, 'P')
    plt.text(drop_off[0]+0.5, drop_off[1]+0.5, 'D')
    for i in range(0,nshooters):
        plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

    p=1
    for s in range(0,o_pick.shape[0]):
        if o_pick[s]==0:
            c = 'c'
        elif o_pick[s]==1:
            c = 'm'
        elif o_pick[s]==2:
            c = 'y'    
        plt.fill([stateSpace[p,1], stateSpace[p,1], stateSpace[p,1]+0.9, stateSpace[p,1]+0.9, stateSpace[p,1]],
                 [stateSpace[p,0], stateSpace[p,0]+0.9, stateSpace[p,0]+0.9, stateSpace[p,0], stateSpace[p,0]],c)
        if p < o.shape[0]:
            p=p+2
            
    plt.savefig(name_drop_off, format='eps')      

    