# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:18:29 2023

@author: seyedmahmouds
"""

import argparse
import numpy as np
import random

def add_args(parser):
    # Add the argument for simulation like net_speed_list
    parser.add_argument('--net_speed_list', type=str, default=[100, 50, 10],
                    metavar='N', help='list of net speeds in mega bytes')
    parser.add_argument('--computation_time_list', type=str, default=[16, 22, 54, 72, 256],
                    metavar='N', help='list of computation time to perform each batch')
    
    args = parser.parse_args()
    return args

parser = argparse.ArgumentParser()
args = add_args(parser)

computation_time_list = list(np.array(args.computation_time_list))
net_speed_list = list(np.array(args.net_speed_list) * 1024 ** 2)
computation_time_weights = [0.1, 0.2, 0.4, 0.2, 0.1]
net_speed_weights = [0.2, 0.3, 0.5]


def agent_simulation(num_agents: int) -> list[list]:# , list:
    '''
    

    Parameters
    ----------
    num_agents : int
        DESCRIPTION.

    Returns
    -------
    ([[net_speeds]] , [computation_speeds])
        DESCRIPTION.

    '''
    
    # Generate net speeds for each pair of agents
    net_speeds = [[random.choices(net_speed_list, weights=net_speed_weights, k = num_agents) for _ in range(num_agents)] for _ in range(num_agents)]


    computation_speeds = [random.choices(computation_time_list, weights=computation_time_weights, k = num_agents) for _ in range(num_agents)]
    
    
    
    return net_speeds, computation_speeds
    
    
    
agent_simulation(4)