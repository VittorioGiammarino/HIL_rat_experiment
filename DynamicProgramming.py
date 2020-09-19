#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:07:10 2020

@author: vittorio
"""

import numpy as np
import StateSpace as ss


def ComputeTransitionProbabilityMatrix(stateSpace,map):

    action_space=5
    K = stateSpace.shape[0]
    P = np.zeros((K,K,action_space))
    [M,N]=map.shape

    for i in range(0,M):
        for j in range(0,N):

            array_temp = [i, j]
            k=ss.FindStateIndex(stateSpace,array_temp)

            if map[i,j] != ss.TREE:

                for u in range(0,action_space):
                    comp_no=1;
                    # east case
                    if j!=N-1:
                        if u == ss.EAST and map[i,j+1]!=ss.TREE:
                            r=i
                            s=j+1
                            comp_no = 0
                        elif j==N-1 and u==ss.EAST:
                            comp_no=1
                    #west case
                    if j!=0:
                        if u==ss.WEST and map[i,j-1]!=ss.TREE:
                            r=i
                            s=j-1
                            comp_no=0
                        elif j==0 and u==ss.WEST:
                            comp_no=1
                    #south case
                    if i!=0:
                        if u==ss.SOUTH and map[i-1,j]!=ss.TREE:
                            r=i-1
                            s=j
                            comp_no=0
                        elif i==0 and u==ss.SOUTH:
                            comp_no=1
                    #north case
                    if i!=M-1:
                        if u==ss.NORTH and map[i+1,j]!=ss.TREE:
                            r=i+1
                            s=j
                            comp_no=0
                        elif i==M-1 and u==ss.NORTH:
                            comp_no=1
                    #hover case
                    if u==ss.HOVER:
                        r=i
                        s=j
                        comp_no=0

                    if comp_no==0:
                        array_temp = [r, s]
                        t=ss.FindStateIndex(stateSpace,array_temp)

                        z=s
                        y=r
                        yy=r+1
                        zz=s+1
                        zzz=s+2
                        yyy=r+2
                        nyy=r-1
                        nyyy=r-2
                        nzz=s-1
                        nzzz=s-2

                    # No wind case
                    #build the probability of being shooted or not

                        p_nh=1

                        if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                            if map[y,z]==ss.SHOOTER:
                                p=ss.GAMMA
                                p_nh=p_nh*(1-p)

                        if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                            if map[y,zz]==ss.SHOOTER:
                                p=ss.GAMMA/(1+1)
                                p_nh=p_nh*(1-p)

                        if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                            if map[y,zzz]==ss.SHOOTER:
                                p=ss.GAMMA/(2+1)
                                p_nh=p_nh*(1-p)

                        if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                            if map[yy,z]==ss.SHOOTER:
                                p=ss.GAMMA/(1+1)
                                p_nh=p_nh*(1-p)

                        if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                            if map[yyy,z]==ss.SHOOTER:
                                p=ss.GAMMA/(1+2)
                                p_nh=p_nh*(1-p)

                        if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                            if map[yy,zz]==ss.SHOOTER:
                                p=ss.GAMMA/(1+2)
                                p_nh=p_nh*(1-p)

                        if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                            if map[y,nzz]==ss.SHOOTER:
                                p=ss.GAMMA/(1+1)
                                p_nh=p_nh*(1-p)
                            
                        if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                            if map[y,nzzz]==ss.SHOOTER:
                                p=ss.GAMMA/(1+2)
                                p_nh=p_nh*(1-p)

                        if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                            if map[nyy,z]==ss.SHOOTER:
                                p=ss.GAMMA/(1+1)
                                p_nh=p_nh*(1-p)

                        if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                            if map[nyyy,z]==ss.SHOOTER:
                                p=ss.GAMMA/(2+1)
                                p_nh=p_nh*(1-p)

                        if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                            if map[nyy,zz]==ss.SHOOTER:
                                p=ss.GAMMA/(2+1)
                                p_nh=p_nh*(1-p)

                        if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                            if map[nyy,nzz]==ss.SHOOTER:
                                p=ss.GAMMA/(2+1)
                                p_nh=p_nh*(1-p)

                        if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                            if map[yy,nzz]==ss.SHOOTER:
                                p=ss.GAMMA/(2+1)
                                p_nh=p_nh*(1-p)
                                
                        P[k,t,u] = P[k,t,u]+(1-ss.P_WIND)*p_nh
                        base0=ss.BaseStateIndex(stateSpace,map)
                        P[k,base0,u]=P[k,base0,u]+(1-ss.P_WIND)*(1-p_nh)

                        # case wind

                        #north wind
                        if s+1>N-1 or map[r,s+1]==ss.TREE:
                            P[k,base0,u]=P[k,base0,u]+ss.P_WIND*0.25 #wind causes crash
                        else:
                            array_temp = [r, s+1]
                            t=ss.FindStateIndex(stateSpace,array_temp)
                            s_nord=s+1

                            z=s_nord
                            y=r
                            yy=r+1
                            zz=s_nord+1
                            zzz=s_nord+2
                            yyy=r+2
                            nzzz=s_nord-2
                            nyy=r-1
                            nyyy=r-2
                            nzz=s_nord-1

                            #build prob of being shot
                            p_nh=1;

                            if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                if map[y,z]==ss.SHOOTER:
                                    p=ss.GAMMA
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                if map[y,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                if map[y,zzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                if map[yy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                if map[yyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                if map[yy,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                if map[y,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                if map[y,nzzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                if map[nyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                if map[nyyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                if map[nyy,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                if map[nyy,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                if map[yy,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)
                                    
                            P[k,t,u] = P[k,t,u]+0.25*ss.P_WIND*p_nh #north wind no hit

                            P[k,base0,u]=P[k,base0,u]+0.25*ss.P_WIND*(1-p_nh) #north wind hit

                        #South Wind
                        if s-1<0 or map[r,s-1]==ss.TREE:
                            P[k,base0,u]=P[k,base0,u]+ss.P_WIND*0.25 #wind causes crash
                        else:
                            array_temp = [r, s-1]
                            t=ss.FindStateIndex(stateSpace,array_temp)
                            s_south=s-1

                            z=s_south
                            y=r
                            yy=r+1
                            zz=s_south+1
                            zzz=s_south+2
                            yyy=r+2
                            nzzz=s_south-2
                            nyy=r-1
                            nyyy=r-2
                            nzz=s_south-1

                            #build prob of being shot
                            p_nh=1;

                            if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                if map[y,z]==ss.SHOOTER:
                                    p=ss.GAMMA
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                if map[y,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                if map[y,zzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                if map[yy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                if map[yyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                if map[yy,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                if map[y,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                if map[y,nzzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                if map[nyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                if map[nyyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                if map[nyy,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                if map[nyy,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                if map[yy,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)
                                    
                            P[k,t,u] = P[k,t,u]+0.25*ss.P_WIND*p_nh #south wind no hit

                            P[k,base0,u]=P[k,base0,u]+0.25*ss.P_WIND*(1-p_nh) #south wind hit

                        #East Wind
                        if r+1>M-1 or map[r+1,s]==ss.TREE:
                            P[k,base0,u]=P[k,base0,u]+ss.P_WIND*0.25 #wind causes crash
                        else:
                            array_temp = [r+1, s]
                            t=ss.FindStateIndex(stateSpace,array_temp)
                            r_east=r+1

                            z=s
                            y=r_east
                            yy=r_east+1
                            zz=s+1
                            zzz=s+2
                            yyy=r_east+2
                            nzzz=s-2
                            nyy=r_east-1
                            nyyy=r_east-2
                            nzz=s-1

                            #build prob of being shot
                            p_nh=1;

                            if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                if map[y,z]==ss.SHOOTER:
                                    p=ss.GAMMA
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                if map[y,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                if map[y,zzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                if map[yy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                if map[yyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                if map[yy,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                if map[y,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                if map[y,nzzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                if map[nyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                if map[nyyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                if map[nyy,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                if map[nyy,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                if map[yy,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)
                                    
                            P[k,t,u] = P[k,t,u]+0.25*ss.P_WIND*p_nh #east wind no hit

                            P[k,base0,u]=P[k,base0,u]+0.25*ss.P_WIND*(1-p_nh) #east wind hit

                        #West Wind
                        if r-1<0 or map[r-1,s]==ss.TREE:
                            P[k,base0,u]=P[k,base0,u]+ss.P_WIND*0.25 #wind causes crash
                        else:
                            array_temp = [r-1, s]
                            t=ss.FindStateIndex(stateSpace,array_temp)
                            r_west=r-1

                            z=s
                            y=r_west
                            yy=r_west+1
                            zz=s+1
                            zzz=s+2
                            yyy=r_west+2
                            nzzz=s-2
                            nyy=r_west-1
                            nyyy=r_west-2
                            nzz=s-1

                            #build prob of being shot
                            p_nh=1;

                            if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                if map[y,z]==ss.SHOOTER:
                                    p=ss.GAMMA
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                if map[y,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                if map[y,zzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                if map[yy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                if map[yyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                if map[yy,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                if map[y,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                if map[y,nzzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+2)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                if map[nyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(1+1)
                                    p_nh=p_nh*(1-p)

                            if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                if map[nyyy,z]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                if map[nyy,zz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                if map[nyy,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)

                            if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                if map[yy,nzz]==ss.SHOOTER:
                                    p=ss.GAMMA/(2+1)
                                    p_nh=p_nh*(1-p)
                                    
                            P[k,t,u] = P[k,t,u]+0.25*ss.P_WIND*p_nh #west wind no hit

                            P[k,base0,u]=P[k,base0,u]+0.25*ss.P_WIND*(1-p_nh) #west wind hit
                    elif comp_no == 1:
                        base0=ss.BaseStateIndex(stateSpace,map)
                        P[k,base0,u]=1

    return P

def ComputeStageCostsR1(stateSpace,map):
        action_space=5
        K = stateSpace.shape[0]
        G = np.zeros((K,action_space))
        [M,N]=map.shape

        for i in range(0,M):
            for j in range(0,N):

                array_temp = [i, j]
                k=ss.FindStateIndex(stateSpace,array_temp)

                if map[i,j] != ss.TREE:

                    if k == ss.R1StateIndex(stateSpace,map):
                        dummy=0 #no cost
                    else:
                        for u in range(0,action_space):
                            comp_no=1;
                            # east case
                            if j!=N-1:
                                if u == ss.EAST and map[i,j+1]!=ss.TREE:
                                    r=i
                                    s=j+1
                                    comp_no = 0
                            elif j==N-1 and u==ss.EAST:
                                comp_no=1

                            if u == ss.EAST:
                                if j==N-1 or map[i,j+1]==ss.TREE:
                                    G[k,u]=np.inf

                            #west case
                            if j!=0:
                                if u==ss.WEST and map[i,j-1]!=ss.TREE:
                                    r=i
                                    s=j-1
                                    comp_no=0
                            elif j==0 and u==ss.WEST:
                                comp_no=1

                            if u==ss.WEST:
                                if j==0 or map[i,j-1]==ss.TREE:
                                    G[k,u]=np.inf

                            #south case
                            if i!=0:
                                if u==ss.SOUTH and map[i-1,j]!=ss.TREE:
                                    r=i-1
                                    s=j
                                    comp_no=0
                            elif i==0 and u==ss.SOUTH:
                                comp_no=1

                            if u==ss.SOUTH:
                                if i==0 or map[i-1,j]==ss.TREE:
                                    G[k,u]=np.inf

                            #north case
                            if i!=M-1:
                                if u==ss.NORTH and map[i+1,j]!=ss.TREE:
                                    r=i+1
                                    s=j
                                    comp_no=0
                            elif i==M-1 and u==ss.NORTH:
                                comp_no=1

                            if u==ss.NORTH:
                                if i==M-1 or map[i+1,j]==ss.TREE:
                                    G[k,u]=np.inf

                            #hover case
                            if u==ss.HOVER:
                                r=i
                                s=j
                                comp_no=0

                            if comp_no==0:
                                array_temp = [r, s]
                                t=ss.FindStateIndex(stateSpace,array_temp)

                                z=s
                                y=r
                                yy=r+1
                                zz=s+1
                                zzz=s+2
                                yyy=r+2
                                nyy=r-1
                                nyyy=r-2
                                nzz=s-1
                                nzzz=s-2

                                #No wind case
                                #build the probability of being shooted or not

                                p_nh=1

                                if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                    if map[y,z]==ss.SHOOTER:
                                        p=ss.GAMMA
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                    if map[y,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                    if map[y,zzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                    if map[yy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                    if map[yyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)

                                if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                    if map[yy,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                    if map[y,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                    if map[y,nzzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)

                                if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                    if map[nyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                    if map[nyyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                    if map[nyy,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                    if map[nyy,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                    if map[yy,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                G[k,u] = G[k,u]+(1-ss.P_WIND)*p_nh*1 #no shot and no wind

                                base0=ss.BaseStateIndex(stateSpace,map)
                                G[k,u] = G[k,u]+(1-ss.P_WIND)*(1-p_nh)*ss.Nc #shot but no wind

                                # case wind

                                #north wind
                                if s+1>N-1 or map[r,s+1]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r, s+1]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    s_nord=s+1

                                    z=s_nord
                                    y=r
                                    yy=r+1
                                    zz=s_nord+1
                                    zzz=s_nord+2
                                    yyy=r+2
                                    nzzz=s_nord-2
                                    nyy=r-1
                                    nyyy=r-2
                                    nzz=s_nord-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #north wind hit

                                #South Wind
                                if s-1<0 or map[r,s-1]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r, s-1]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    s_south=s-1

                                    z=s_south
                                    y=r
                                    yy=r+1
                                    zz=s_south+1
                                    zzz=s_south+2
                                    yyy=r+2
                                    nzzz=s_south-2
                                    nyy=r-1
                                    nyyy=r-2
                                    nzz=s_south-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1 #south wind no hit

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #south wind hit

                                #East Wind
                                if r+1>M-1 or map[r+1,s]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r+1, s]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    r_east=r+1

                                    z=s
                                    y=r_east
                                    yy=r_east+1
                                    zz=s+1
                                    zzz=s+2
                                    yyy=r_east+2
                                    nzzz=s-2
                                    nyy=r_east-1
                                    nyyy=r_east-2
                                    nzz=s-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1 #east wind no hit

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #east wind hit

                                #West Wind
                                if r-1<0 or map[r-1,s]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r-1, s]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    r_west=r-1

                                    z=s
                                    y=r_west
                                    yy=r_west+1
                                    zz=s+1
                                    zzz=s+2
                                    yyy=r_west+2
                                    nzzz=s-2
                                    nyy=r_west-1
                                    nyyy=r_west-2
                                    nzz=s-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1 #west wind no hit

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #east wind hit
                            elif comp_no == 1:
                                dummy=0

        k = ss.R1StateIndex(stateSpace, map)

        for l in range(0,action_space):
            G[k,l]=0

        return G
    
def ComputeStageCostsR2(stateSpace,map):
        action_space=5
        K = stateSpace.shape[0]
        G = np.zeros((K,action_space))
        [M,N]=map.shape

        for i in range(0,M):
            for j in range(0,N):

                array_temp = [i, j]
                k=ss.FindStateIndex(stateSpace,array_temp)

                if map[i,j] != ss.TREE:

                    if k == ss.R2StateIndex(stateSpace,map):
                        dummy=0 #no cost
                    else:
                        for u in range(0,action_space):
                            comp_no=1;
                            # east case
                            if j!=N-1:
                                if u == ss.EAST and map[i,j+1]!=ss.TREE:
                                    r=i
                                    s=j+1
                                    comp_no = 0
                            elif j==N-1 and u==ss.EAST:
                                comp_no=1

                            if u == ss.EAST:
                                if j==N-1 or map[i,j+1]==ss.TREE:
                                    G[k,u]=np.inf

                            #west case
                            if j!=0:
                                if u==ss.WEST and map[i,j-1]!=ss.TREE:
                                    r=i
                                    s=j-1
                                    comp_no=0
                            elif j==0 and u==ss.WEST:
                                comp_no=1

                            if u==ss.WEST:
                                if j==0 or map[i,j-1]==ss.TREE:
                                    G[k,u]=np.inf

                            #south case
                            if i!=0:
                                if u==ss.SOUTH and map[i-1,j]!=ss.TREE:
                                    r=i-1
                                    s=j
                                    comp_no=0
                            elif i==0 and u==ss.SOUTH:
                                comp_no=1

                            if u==ss.SOUTH:
                                if i==0 or map[i-1,j]==ss.TREE:
                                    G[k,u]=np.inf

                            #north case
                            if i!=M-1:
                                if u==ss.NORTH and map[i+1,j]!=ss.TREE:
                                    r=i+1
                                    s=j
                                    comp_no=0
                            elif i==M-1 and u==ss.NORTH:
                                comp_no=1

                            if u==ss.NORTH:
                                if i==M-1 or map[i+1,j]==ss.TREE:
                                    G[k,u]=np.inf

                            #hover case
                            if u==ss.HOVER:
                                r=i
                                s=j
                                comp_no=0

                            if comp_no==0:
                                array_temp = [r, s]
                                t=ss.FindStateIndex(stateSpace,array_temp)

                                z=s
                                y=r
                                yy=r+1
                                zz=s+1
                                zzz=s+2
                                yyy=r+2
                                nyy=r-1
                                nyyy=r-2
                                nzz=s-1
                                nzzz=s-2

                                #No wind case
                                #build the probability of being shooted or not

                                p_nh=1

                                if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                    if map[y,z]==ss.SHOOTER:
                                        p=ss.GAMMA
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                    if map[y,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                    if map[y,zzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                    if map[yy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                    if map[yyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)

                                if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                    if map[yy,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                    if map[y,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                    if map[y,nzzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)

                                if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                    if map[nyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                    if map[nyyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                    if map[nyy,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                    if map[nyy,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                    if map[yy,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                G[k,u] = G[k,u]+(1-ss.P_WIND)*p_nh*1 #no shot and no wind

                                base0=ss.BaseStateIndex(stateSpace,map)
                                G[k,u] = G[k,u]+(1-ss.P_WIND)*(1-p_nh)*ss.Nc #shot but no wind

                                # case wind

                                #north wind
                                if s+1>N-1 or map[r,s+1]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r, s+1]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    s_nord=s+1

                                    z=s_nord
                                    y=r
                                    yy=r+1
                                    zz=s_nord+1
                                    zzz=s_nord+2
                                    yyy=r+2
                                    nzzz=s_nord-2
                                    nyy=r-1
                                    nyyy=r-2
                                    nzz=s_nord-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #north wind hit

                                #South Wind
                                if s-1<0 or map[r,s-1]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r, s-1]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    s_south=s-1

                                    z=s_south
                                    y=r
                                    yy=r+1
                                    zz=s_south+1
                                    zzz=s_south+2
                                    yyy=r+2
                                    nzzz=s_south-2
                                    nyy=r-1
                                    nyyy=r-2
                                    nzz=s_south-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1 #south wind no hit

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #south wind hit

                                #East Wind
                                if r+1>M-1 or map[r+1,s]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r+1, s]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    r_east=r+1

                                    z=s
                                    y=r_east
                                    yy=r_east+1
                                    zz=s+1
                                    zzz=s+2
                                    yyy=r_east+2
                                    nzzz=s-2
                                    nyy=r_east-1
                                    nyyy=r_east-2
                                    nzz=s-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1 #east wind no hit

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #east wind hit

                                #West Wind
                                if r-1<0 or map[r-1,s]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r-1, s]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    r_west=r-1

                                    z=s
                                    y=r_west
                                    yy=r_west+1
                                    zz=s+1
                                    zzz=s+2
                                    yyy=r_west+2
                                    nzzz=s-2
                                    nyy=r_west-1
                                    nyyy=r_west-2
                                    nzz=s-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1 #west wind no hit

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #east wind hit
                            elif comp_no == 1:
                                dummy=0

        k = ss.R2StateIndex(stateSpace, map)

        for l in range(0,action_space):
            G[k,l]=0

        return G
    
def ComputeStageCostsR1andR2(stateSpace,map):
        action_space=5
        K = stateSpace.shape[0]
        G = np.zeros((K,action_space))
        [M,N]=map.shape

        for i in range(0,M):
            for j in range(0,N):

                array_temp = [i, j]
                k=ss.FindStateIndex(stateSpace,array_temp)

                if map[i,j] != ss.TREE:

                    if k == ss.R1StateIndex(stateSpace,map):
                        dummy=0 #no cost
                    elif k == ss.R2StateIndex(stateSpace,map):
                        dummy = 0
                    else:
                        for u in range(0,action_space):
                            comp_no=1;
                            # east case
                            if j!=N-1:
                                if u == ss.EAST and map[i,j+1]!=ss.TREE:
                                    r=i
                                    s=j+1
                                    comp_no = 0
                            elif j==N-1 and u==ss.EAST:
                                comp_no=1

                            if u == ss.EAST:
                                if j==N-1 or map[i,j+1]==ss.TREE:
                                    G[k,u]=np.inf

                            #west case
                            if j!=0:
                                if u==ss.WEST and map[i,j-1]!=ss.TREE:
                                    r=i
                                    s=j-1
                                    comp_no=0
                            elif j==0 and u==ss.WEST:
                                comp_no=1

                            if u==ss.WEST:
                                if j==0 or map[i,j-1]==ss.TREE:
                                    G[k,u]=np.inf

                            #south case
                            if i!=0:
                                if u==ss.SOUTH and map[i-1,j]!=ss.TREE:
                                    r=i-1
                                    s=j
                                    comp_no=0
                            elif i==0 and u==ss.SOUTH:
                                comp_no=1

                            if u==ss.SOUTH:
                                if i==0 or map[i-1,j]==ss.TREE:
                                    G[k,u]=np.inf

                            #north case
                            if i!=M-1:
                                if u==ss.NORTH and map[i+1,j]!=ss.TREE:
                                    r=i+1
                                    s=j
                                    comp_no=0
                            elif i==M-1 and u==ss.NORTH:
                                comp_no=1

                            if u==ss.NORTH:
                                if i==M-1 or map[i+1,j]==ss.TREE:
                                    G[k,u]=np.inf

                            #hover case
                            if u==ss.HOVER:
                                r=i
                                s=j
                                comp_no=0

                            if comp_no==0:
                                array_temp = [r, s]
                                t=ss.FindStateIndex(stateSpace,array_temp)

                                z=s
                                y=r
                                yy=r+1
                                zz=s+1
                                zzz=s+2
                                yyy=r+2
                                nyy=r-1
                                nyyy=r-2
                                nzz=s-1
                                nzzz=s-2

                                #No wind case
                                #build the probability of being shooted or not

                                p_nh=1

                                if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                    if map[y,z]==ss.SHOOTER:
                                        p=ss.GAMMA
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                    if map[y,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                    if map[y,zzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                    if map[yy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                    if map[yyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)

                                if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                    if map[yy,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                    if map[y,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                    if map[y,nzzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+2)
                                        p_nh=p_nh*(1-p)

                                if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                    if map[nyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(1+1)
                                        p_nh=p_nh*(1-p)

                                if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                    if map[nyyy,z]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                    if map[nyy,zz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                    if map[nyy,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                    if map[yy,nzz]==ss.SHOOTER:
                                        p=ss.GAMMA/(2+1)
                                        p_nh=p_nh*(1-p)

                                G[k,u] = G[k,u]+(1-ss.P_WIND)*p_nh*1 #no shot and no wind

                                base0=ss.BaseStateIndex(stateSpace,map)
                                G[k,u] = G[k,u]+(1-ss.P_WIND)*(1-p_nh)*ss.Nc #shot but no wind

                                # case wind

                                #north wind
                                if s+1>N-1 or map[r,s+1]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r, s+1]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    s_nord=s+1

                                    z=s_nord
                                    y=r
                                    yy=r+1
                                    zz=s_nord+1
                                    zzz=s_nord+2
                                    yyy=r+2
                                    nzzz=s_nord-2
                                    nyy=r-1
                                    nyyy=r-2
                                    nzz=s_nord-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #north wind hit

                                #South Wind
                                if s-1<0 or map[r,s-1]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r, s-1]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    s_south=s-1

                                    z=s_south
                                    y=r
                                    yy=r+1
                                    zz=s_south+1
                                    zzz=s_south+2
                                    yyy=r+2
                                    nzzz=s_south-2
                                    nyy=r-1
                                    nyyy=r-2
                                    nzz=s_south-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1 #south wind no hit

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #south wind hit

                                #East Wind
                                if r+1>M-1 or map[r+1,s]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r+1, s]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    r_east=r+1

                                    z=s
                                    y=r_east
                                    yy=r_east+1
                                    zz=s+1
                                    zzz=s+2
                                    yyy=r_east+2
                                    nzzz=s-2
                                    nyy=r_east-1
                                    nyyy=r_east-2
                                    nzz=s-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1 #east wind no hit

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #east wind hit

                                #West Wind
                                if r-1<0 or map[r-1,s]==ss.TREE:
                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*ss.Nc #wind causes crash
                                else:
                                    array_temp = [r-1, s]
                                    t=ss.FindStateIndex(stateSpace,array_temp)
                                    r_west=r-1

                                    z=s
                                    y=r_west
                                    yy=r_west+1
                                    zz=s+1
                                    zzz=s+2
                                    yyy=r_west+2
                                    nzzz=s-2
                                    nyy=r_west-1
                                    nyyy=r_west-2
                                    nzz=s-1

                                    #build prob of being shot
                                    p_nh=1;

                                    if y<=M-1 and z<=N-1 and y>=0 and z>=0:
                                        if map[y,z]==ss.SHOOTER:
                                            p=ss.GAMMA
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zz<=N-1 and y>=0 and zz>=0:
                                        if map[y,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and zzz<=N-1 and y>=0 and zzz>=0:
                                        if map[y,zzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and z<=N-1 and yy>=0 and z>=0:
                                        if map[yy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if yyy<=M-1 and z<=N-1 and yyy>=0 and z>=0:
                                        if map[yyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and zz<=N-1 and yy>=0 and zz>=0:
                                        if map[yy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzz<=N-1 and y>=0 and nzz>=0:
                                        if map[y,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if y<=M-1 and nzzz<=N-1 and y>=0 and nzzz>=0:
                                        if map[y,nzzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+2)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and z<=N-1 and nyy>=0 and z>=0:
                                        if map[nyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(1+1)
                                            p_nh=p_nh*(1-p)

                                    if nyyy<=M-1 and z<=N-1 and nyyy>=0 and z>=0:
                                        if map[nyyy,z]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and zz<=N-1 and nyy>=0 and zz>=0:
                                        if map[nyy,zz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if nyy<=M-1 and nzz<=N-1 and nyy>=0 and nzz>=0:
                                        if map[nyy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    if yy<=M-1 and nzz<=N-1 and yy>=0 and nzz>=0:
                                        if map[yy,nzz]==ss.SHOOTER:
                                            p=ss.GAMMA/(2+1)
                                            p_nh=p_nh*(1-p)

                                    G[k,u]=G[k,u]+ss.P_WIND*0.25*p_nh*1 #west wind no hit

                                    G[k,u]=G[k,u]+0.25*ss.P_WIND*(1-p_nh)*ss.Nc #east wind hit
                            elif comp_no == 1:
                                dummy=0

        k = ss.R1StateIndex(stateSpace, map)

        for l in range(0,action_space):
            G[k,l]=0
            
        k = ss.R2StateIndex(stateSpace, map)

        for l in range(0,action_space):
            G[k,l]=0

        return G

def ValueIteration(P,G,TERMINAL_INDEX):

    action_space=5
    tol=10**(-5)
    K = G.shape[0]
    V=np.zeros((K,action_space))
    VV=np.zeros((K,2))
    I=np.zeros((K))
    Err=np.zeros((K))

    #initialization
    VV[:,0]=50
    VV[TERMINAL_INDEX,0]=0
    n=0
    Check_err=1

    while Check_err==1:
        n=n+1
        Check_err=0
        for k in range(0,K):
            if n>1:
                VV[:,0]=VV[0:,1]

            if k==TERMINAL_INDEX:
                VV[k,1]=0
                V[k,:]=0
            else:
                CTG=np.zeros((action_space)) #cost to go
                for u in range(0,action_space):
                    for j in range(0,K):
                        CTG[u]=CTG[u] + P[k,j,u]*VV[j,1]

                    V[k,u]=G[k,u]+CTG[u]

                VV[k,1]=np.amin(V[k,:])
                flag = np.where(V[k,:]==np.amin(V[k,:]))
                I[k]=flag[0][0]

            Err[k]=abs(VV[k,1]-VV[k,0])

            if Err[k]>tol:
                Check_err=1

    J_opt=VV[:,1]
    I[TERMINAL_INDEX]=ss.HOVER
    u_opt = I[:]

    return J_opt,u_opt

def ValueIteration_Both(P,G,TERMINAL_INDEX1, TERMINAL_INDEX2):

    action_space=5
    tol=10**(-5)
    K = G.shape[0]
    V=np.zeros((K,action_space))
    VV=np.zeros((K,2))
    I=np.zeros((K))
    Err=np.zeros((K))

    #initialization
    VV[:,0]=50
    VV[TERMINAL_INDEX1,0]=0
    VV[TERMINAL_INDEX2,0]=0
    n=0
    Check_err=1

    while Check_err==1:
        n=n+1
        Check_err=0
        for k in range(0,K):
            if n>1:
                VV[:,0]=VV[0:,1]

            if k==TERMINAL_INDEX1:
                VV[k,1]=0
                V[k,:]=0
            elif k==TERMINAL_INDEX2:
                VV[k,1]=0
                V[k,:]=0
            else:
                CTG=np.zeros((action_space)) #cost to go
                for u in range(0,action_space):
                    for j in range(0,K):
                        CTG[u]=CTG[u] + P[k,j,u]*VV[j,1]

                    V[k,u]=G[k,u]+CTG[u]

                VV[k,1]=np.amin(V[k,:])
                flag = np.where(V[k,:]==np.amin(V[k,:]))
                I[k]=flag[0][0]

            Err[k]=abs(VV[k,1]-VV[k,0])

            if Err[k]>tol:
                Check_err=1

    J_opt=VV[:,1]
    I[TERMINAL_INDEX1]=ss.HOVER
    I[TERMINAL_INDEX2]=ss.HOVER
    u_opt = I[:]

    return J_opt,u_opt
