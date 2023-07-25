# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:18:29 2023

@author: seyedmahmouds
"""

import numpy as np

def AgentRemainingTime_check_others(j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_tier_size):
    '''
    this function check all other agents to see with is the best to pair with

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
            tau_m_jk = remaining_batch_numbers[j] * batch_data_tier_size[m] / net_speeds[j][k] # feature_map_size is size of all remaning data

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


def AgentRemainingTime(i, j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_tier_size, slow_fast_tier_training_time):
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
    i : slow agent which offload the model
    j : the paired fast agent

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
    
    tau_wait_i = remaining_times[i] 
    tau_wait_j = remaining_times[j] # wait time until client complete its task
    
    tau_i_list = []
    tier_ij_list = []


    # Ask agent k for its speed and remaining time to complete its current training task
    
    
    tau_m_i_list = []
    p_m_i, p_m_j = 0, 0

    for m in range(num_tiers):
        # Estimate tau_m_jk (estimated communication time for tier m between agent j and k)
        # based on communication speed observed between agent j and k
        tau_m_ij = remaining_batch_numbers[i] * (np.array(batch_data_tier_size[m]) / np.array(net_speeds[i][j])) # to consider inf
         
        # Estimate tau_m_j and tau_m_k (estimated time for tier m for agent j)
        # tau_m_j = remaining_feature_map_size[j] / computation_speeds[j]
        
        if j != i and m != 0:
            p_m_i = computation_speeds[i] / slow_fast_tier_training_time[m][0] # time to done each batch of data
            p_m_j = computation_speeds[j] / slow_fast_tier_training_time[m][1] # 0 for slow side, 1 for fast side
        
        if m == 0 or i == j:
            tau_m_i_list.append(tau_wait_i) # tau_wait_i should be = to remaining_batch_numbers[j] / max(tau_m_j , tau_m_ij , tau_m_i) )
        # elif max(tau_m_j , tau_m_ij , tau_m_i) == np.inf:
        #     tau_m_i_list.append(np.inf)
        else:
            tau_m_i = max(remaining_batch_numbers[i] / p_m_i, remaining_batch_numbers[i] / p_m_j + tau_wait_j, tau_m_ij)
            tau_m_i_list.append(tau_m_i)
    
    tau_i_list.append(min(tau_m_i_list)) # so far I have not considered if the this tier selection makes the fast agent time bigger than this slow agent
    # print(tau_m_i_list)
    tier_ij_list.append(tau_m_i_list.index(tau_i_list[-1]))
        
    tau_j = min(tau_i_list)
    # tier_j = tier_ij_list.index(tau_j)
    tier_j = tier_ij_list[0]

    return tau_j, tier_j




def AgentTrainingTime(i, j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_tier_size, slow_fast_tier_training_time):
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
    i : slow agent which offload the model
    j : the paired fast agent

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
    
    tau_wait_i = remaining_times[i] 
    tau_wait_j = remaining_times[j] # wait time until client complete its task
    
    tau_i_list = []
    tier_ij_list = []


    # Ask agent k for its speed and remaining time to complete its current training task
    
    
    tau_m_i_list = []
    p_m_i, p_m_j = 0, 0

    for m in range(num_tiers):
        # Estimate tau_m_jk (estimated communication time for tier m between agent j and k)
        # based on communication speed observed between agent j and k
        tau_m_ij = remaining_batch_numbers[i] * (np.array(batch_data_tier_size[m]) / np.array(net_speeds[i][j])) # to consider inf
         
        # Estimate tau_m_j and tau_m_k (estimated time for tier m for agent j)
        # tau_m_j = remaining_feature_map_size[j] / computation_speeds[j]
        
        if j != i and m != 0:
            p_m_i = computation_speeds[i] / slow_fast_tier_training_time[m][0] # time to done each batch of data
            p_m_j = computation_speeds[j] / slow_fast_tier_training_time[m][1] # 0 for slow side, 1 for fast side
        
        if m == 0 or i == j:
            tau_m_i_list.append(tau_wait_i) # tau_wait_i should be = to remaining_batch_numbers[j] / max(tau_m_j , tau_m_ij , tau_m_i) )
        # elif max(tau_m_j , tau_m_ij , tau_m_i) == np.inf:
        #     tau_m_i_list.append(np.inf)
        else:
            tau_m_i = max(remaining_batch_numbers[i] / p_m_i, remaining_batch_numbers[i] / p_m_j + tau_wait_j, tau_m_ij)
            tau_m_i_list.append(tau_m_i)
    
    tau_i_list.append(min(tau_m_i_list)) # so far I have not considered if the this tier selection makes the fast agent time bigger than this slow agent
    # print(tau_m_i_list)
    tier_ij_list.append(tau_m_i_list.index(tau_i_list[-1]))
        
    tau_j = min(tau_i_list) # list to num
    # tier_j = tier_ij_list.index(tau_j)
    tier_j = tier_ij_list[0] # list to num

    return tau_j, tier_j


# this part is only for testing the scheduler

# ''' 
net_speeds = [[np.inf, 50, 50, 20, 0],
              [50, np.inf, 50, 10, 10],
              [50, 50, np.inf, 50, 10],
              [50, 10, 50, np.inf, 0],
              [0, 10, 10, 0, np.inf]] # mbps
computation_speeds = [0.4, 1, 1.2, 0.2, 0.8] # batch per second
remaining_batch_numbers = [12, 10, 15, 10, 12]
remaining_times = list(np.array(remaining_batch_numbers) / np.array(computation_speeds)) # if continue without offloading
agents_list = [0, 1, 2, 3, 4]
num_agents, i, j = 5, 1, 3 # client j offloads to agent i

# net_speeds = [[np.inf, 50],
#               [50, np.inf]] # mbps
# computation_speeds = [0.4, 1] # batch per second
# remaining_batch_numbers = [12, 10]
# remaining_times = list(np.array(remaining_batch_numbers) / np.array(computation_speeds)) # if continue without offloading


# num_agents, i, j = 2, 1, 0 # client j offloads to agent i
# agents_list = [0, 1]

# parameters related to the split layer
# num_tiers = 4
TIER_ZERO = 0
# batch_data_tier_size = [0, 20, 35, 30]
# slow_fast_tier_training_time = [[1, 0], [0.8, 0.4], [0.6, 0.6], [0.2, 0.9]]

 
# tau_j, tier_j = AgentRemainingTime(i, j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, len(agents_list), batch_data_tier_size, slow_fast_tier_training_time)
# tau_j, tier_j = AgentRemainingTime_check_others(j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_tier_size)
# '''


# this function is the Pairing function in the paper
def pairing_scheduler(i, net_speeds, computation_speeds, remaining_times, num_tiers,
                      remaining_batch_numbers, num_agents, batch_data_tier_size, 
                      slow_fast_tier_training_time, available_agents_list):
    
    tau, tier = [], []
    
    for j in range(num_agents):
        if j in available_agents_list:
            tau_j, tier_j = \
                AgentTrainingTime(i, j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, num_agents, batch_data_tier_size, slow_fast_tier_training_time)
        else:
            tau_j, tier_j = np.inf, TIER_ZERO
        tau.append(tau_j)
        tier.append(tier_j)
        
    tau_i = min(tau)
    paired_j = tau.index(tau_i)
    tier_i = tier[paired_j]
        
    
    return tau_i, tier_i, paired_j

def pairing(agents_ordered_list : list,
            net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, batch_data_tier_size,
            slow_fast_tier_training_time) -> list[list]:
    '''
    pairing of all agents at the beggining of each epoch

    Parameters
    ----------
    agents_ordered_list : list
        DESCRIPTION.

    Returns
    -------
    list[list]
        DESCRIPTION.

    '''
    
    num_agents = len(agents_ordered_list)
    available_agents_list = list(range(0,num_agents))
    paired_list = {}
    tier_list = {}
    tau_i_list = []
    for i in agents_ordered_list:
        
        if i not in paired_list.keys() and i not in paired_list.values():
            tau_i, tier_i, j = pairing_scheduler(i, net_speeds, computation_speeds, remaining_times, num_tiers,
                                                 remaining_batch_numbers, num_agents, batch_data_tier_size,
                                                 slow_fast_tier_training_time, available_agents_list)
            # paired_list.append(i)
            
            tier_list[i] = tier_i
            tau_i_list.append(tau_i) # just to keep track of times, not true for agent with tier 0
            if tier_i != TIER_ZERO:
                paired_list[i] = j
                tier_list[j] = TIER_ZERO
                available_agents_list.pop(available_agents_list.index(i))
                available_agents_list.pop(available_agents_list.index(j))
            # agents_ordered_list = np.delete(agents_ordered_list, [i,j])
            
    return paired_list, tier_list, tau_i_list
            


''' for test

# agents_ordered_list = agents_list
agents_ordered_list = np.argsort(remaining_times)[::-1]
            
paired_list, tier_list = pairing(agents_ordered_list, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, batch_data_tier_size, slow_fast_tier_training_time)



paired_list = {}
remaining_times_initial = remaining_times.copy()
remaining_times_with_pairing = remaining_times.copy()
completed_batch_numbers = remaining_batch_numbers.copy()


## this while can be used for the version 1 of algorithm which fastest agent ask for the remaining time of others

while len(agents_list) > 1:
    index_agent_complete = remaining_times_initial.index(min(remaining_times))

    i = index_agent_complete
    # i = 0
    agents_remaining_time_paired = {}
    agents_paired = {}
    for j in agents_list:
        tau_j, tier_j = AgentRemainingTime(i, j, net_speeds, computation_speeds, remaining_times_initial, num_tiers, remaining_batch_numbers, len(agents_list), batch_data_tier_size, slow_fast_tier_training_time)
        agents_remaining_time_paired[j] = tau_j
        agents_paired[j] = tier_j
        # max(agents_remaining_time_paired.values())
        # print(f'i={i}, j={j}, tau_j={tau_j}, tier_j={tier_j}')
    j = max(agents_remaining_time_paired, key=agents_remaining_time_paired.get)
    remaining_times_with_pairing[j] = max(agents_remaining_time_paired.values())
    agents_list.remove(i), agents_list.remove(j)
    # remaining_times.pop(i), remaining_times.pop(j)
    remaining_times.remove(remaining_times_initial[i])
    remaining_times.remove(remaining_times_initial[j])
    paired_list[i] = j
    
    # calculate number of batches before pairing
    completed_batch_numbers[j] = int(remaining_times_initial[i] / remaining_times_initial[j] * completed_batch_numbers[j])
    
## this while can be used for the version 2 of algorithm, where slowest agent first select its pair and continue


agents_list = [0, 1, 2, 3, 4]

paired_list = {}
remaining_times = remaining_times_initial.copy()
remaining_times_with_pairing = remaining_times_initial.copy()
remaining_batch_numbers = completed_batch_numbers.copy()

while len(agents_list) > 1: # i and j is not same as paper in this part
    index_slowest_agent = remaining_times_initial.index(max(remaining_times))

    i = index_slowest_agent
    agents_remaining_time_paired = {}
    agents_paired = {}
    for j in agents_list:
        tau_j, tier_j = AgentRemainingTime(j, i, net_speeds, computation_speeds, remaining_times_initial, num_tiers, remaining_batch_numbers, len(agents_list), batch_data_tier_size, slow_fast_tier_training_time)
        agents_remaining_time_paired[j] = tau_j
        agents_paired[j] = tier_j
        # max(agents_remaining_time_paired.values())
        # print(f'i={i}, j={j}, tau_j={tau_j}, tier_j={tier_j}')
    j = max(agents_remaining_time_paired, key=agents_remaining_time_paired.get)
    remaining_times_with_pairing[j] = max(agents_remaining_time_paired.values())
    agents_list.remove(i), agents_list.remove(j)
    # remaining_times.pop(i), remaining_times.pop(j)
    remaining_times.remove(remaining_times_initial[i])
    remaining_times.remove(remaining_times_initial[j])
    paired_list[i] = j
    
    # I can change it to see how much percentage of client should allocate to other one
    # completed_batch_numbers[j] = int(remaining_times_initial[i] / remaining_times_initial[j] * completed_batch_numbers[j])
    
'''   
    
    
'''
for i in agents_list: # i help agent j
    for j in agents_list:
        tau_j, tier_j = AgentRemainingTime(i, j, net_speeds, computation_speeds, remaining_times, num_tiers, remaining_batch_numbers, len(agents_list), batch_data_tier_size, slow_fast_tier_training_time)
        print(f'i={i}, j={j}, tau_j={tau_j}, tier_j={tier_j}')
        
        
'''