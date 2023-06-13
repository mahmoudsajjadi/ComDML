# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:18:29 2023

@author: seyedmahmouds
"""

import numpy as np

def AgentRemainingTime(j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_size):
    '''
    

    Parameters
    ----------
    j : TYPE
        DESCRIPTION.
    net_speeds : TYPE
        DESCRIPTION.
    remaining_times : TYPE
        DESCRIPTION.
    num_tiers : TYPE
        DESCRIPTION.
    computation_speeds:
        spped to fullfil one batch

    Returns
    -------
    tau_j : TYPE
        DESCRIPTION.
        
    remaining time has 4 elements:
        1. wait time until client complete its task (tau_wait_k)
        2. communicatio time
        3. processing time of slow clinet to complete first batch
        4. processing time of fast clinet to complete last batch

    '''
    
    # Estimate remaining time if agent j begin training collaboratively
    # tau_j = remaining_times[j]
    tau_wait_j = remaining_times[j]
    
    tau_m_j_list = []
    tier_jk = []

    for k in range(num_agents): # change this to check all other agents, just make inf if they are not connected
        # Ask agent k for its speed and remaining time to complete its current training task
        tau_wait_k = remaining_times[k] # wait time until client complete its task
        
        tau_m_jk_list = []
        tau_m_j, tau_m_k = 0, 0

        for m in range(num_tiers):
            # Estimate tau_m_jk (estimated communication time for tier m between agent j and k)
            # based on communication speed observed between agent j and k
            tau_m_jk = remaining_batch_numbers[j] * batch_data_size[m] / net_speeds[j][k] # feature_map_size is size of all remaning data

            # Estimate tau_m_j and tau_m_k (estimated time for tier m for agent j)
            # tau_m_j = remaining_feature_map_size[j] / computation_speeds[j]
            
            if j != k and remaining_batch_numbers[j] > 0:
                tau_m_j = computation_speeds[j] # time to done each batch of data
                tau_m_k = computation_speeds[k]
            
            tau_m_jk_list.append(max(tau_wait_j, tau_wait_k) + tau_m_j + tau_m_jk + tau_m_k)
        
        tau_m_j_list.append(min(tau_m_jk_list)) # so far I have not considered if the this tier selection makes the fast agent time bigger than this slow agent
        tier_jk.append(tau_m_jk_list.index(tau_m_j_list[-1]))
        
    tau_j = min(tau_m_j_list)
    tier_j = tier_jk.index(tau_j)

    return tau_j, tier_j

net_speeds = [[np.inf, 50, 0, 20, 0],
              [50, np.inf, 50, 10, 10],
              [0, 50, np.inf, 50, 10],
              [50, 10, 50, np.inf, 0],
              [0, 10, 10, 0, np.inf]] # mbps
computation_speeds = [0.4, 1, 1.2, 0.2, 0.8] # batch per second
remaining_batch_numbers = [12, 10, 15, 10, 12]

remaining_times = list(np.array(remaining_batch_numbers) / np.array(computation_speeds)) # if continue without offloading


j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_size = \
    0, [[10,10],[20,10]], [4,2], [0,20],  2, [0, 8], 2, [0, 300]
tau_j, tier_j = AgentRemainingTime(j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_size)


def pairing_scheduler(i, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_size):
    
    tau, tier = [], []
    
    for j in range(num_agents):
        tau_j, tier_j = \
            AgentRemainingTime(j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_size)
        tau.append(tau_j)
        tier.append(tier_j)
        
    tau = max(tau_j)
    tier = tau_j.index(tau)
        
    
    return tau, tier