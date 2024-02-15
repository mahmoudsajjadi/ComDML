# -*- coding: utf-8 -*-
def AgentRemainingTime(j, neighbors, speeds, remaining_times):
    # Estimate remaining time if agent j begin training collaboratively
    tau_j = remaining_times[j]

    for k in neighbors[j]: # change this 
        # Ask agent k for its speed and remaining time to complete its current training task
        speed_k = speeds[k]
        tau_wait_k = remaining_times[k]

        for m in range(num_tiers):
            # Estimate tau_m_jk (estimated time for tier m for agent j and agent k)
            # based on communication speed observed between agent j and agent k
            tau_m_jk = estimate_tau_m_jk(speed_j, speed_k, tau_j)

            # Estimate tau_m_j (estimated time for tier m for agent j)
            tau_m_j = estimate_tau_m_j(tau_wait_k, tau_m_jk, tau_m_j)

        tau_jk = min(tau_m_jk for tau_m_jk in tau_m_jk_list)
        tau_j = min(tau_jk, tau_j)

    return tau_j
