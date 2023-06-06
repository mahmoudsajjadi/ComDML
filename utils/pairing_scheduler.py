# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:18:29 2023

@author: seyedmahmouds
"""

def AgentRemainingTime(j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_feature_map_size, num_agents):
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
    tau_j = remaining_times[j]
    
    tau_m_j_list = []
    tier_jk = []

    for k in range(num_agents): # change this to check all other agents, just make inf if they are not connected
        # Ask agent k for its speed and remaining time to complete its current training task
        tau_wait_k = remaining_times[k] # wait time until client complete its task
        
        tau_m_jk_list = []

        for m in range(num_tiers):
            # Estimate tau_m_jk (estimated communication time for tier m between agent j and k)
            # based on communication speed observed between agent j and k
            tau_m_jk = remaining_feature_map_size[j] / net_speeds[j][k] # feature_map_size is size of all remaning data

            # Estimate tau_m_j and tau_m_k (estimated time for tier m for agent j)
            # tau_m_j = remaining_feature_map_size[j] / computation_speeds[j]
            tau_m_j = 1 / computation_speeds[j]
            tau_m_k = 1 / computation_speeds[k]
            
            tau_m_jk_list.append(tau_m_j + tau_m_jk + tau_m_k)
        
        tau_m_j_list.append(min(tau_m_jk_list))
        tier_jk.append(tau_m_jk_list.index(tau_m_j_list[-1]))
        
    tau_j = min(tau_m_j_list)
    tier_jk

    return tau_j
