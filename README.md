## Communication-Efficient Training Workload Balancing for Distributed Multi-Agent Learning
### Fully Distributed Split Federated Learning

This repository contains the Python implementation of ComDML, a novel approach for Communication-efficient Training Workload Balancing in Decentralized Multi-Agent Learning. ComDML tackles the challenge of uneven training workloads in decentralized systems, facilitating efficient model training without relying on a central server. By enabling agents to collaborate and share the training burden, ComDML empowers scalable and resource-optimized DML applications. The paper describing ComDML is going to be published at the IEEE International Conference on Distributed Computing Systems (ICDCS) 2024, in New Jersey.

The training workflow for ComDML is illustrated here:
![Training Process](https://github.com/mahmoudsajjadi/ComDML/blob/main/training_process.png)


### Watch the Presentation
To get an in-depth understanding of ComDML, watch the presentation from the ICDCS 2024 conference:
[![Watch the video](https://img.youtube.com/vi/62QmwyW5tqc/0.jpg)](https://youtu.be/62QmwyW5tqc)

The presentation file is available [here](https://github.com/mahmoudsajjadi/ComDML/blob/main/ComDML%20-%20Online%20.pptx).




### Implementation
This repository includes the necessary code to replicate experiments and run ComDML on various datasets and models. To run ComDML, ensure you have the following dependencies installed:

- Python
- PyTorch
- NumPy
- torchvision
- pandas
- scikit-learn
- matplotlib
- wandb

To install the required Python packages, you can use pip:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib wandb
```

### Abstract
Decentralized Multi-agent Learning (DML) enables collaborative model training while preserving data privacy. However, inherent heterogeneity in agents’ resources may lead to substantial variations in training times, causing straggler effects and resource wastage. To address this, ComDML balances workload among agents through decentralized workload offloading, leveraging local-loss split training. This approach optimizes workload balancing by considering both communication and computation capacities of agents. A dynamic decentralized pairing scheduler efficiently pairs agents and determines optimal offloading amounts. ComDML demonstrates robustness in heterogeneous environments, significantly reducing training time while maintaining model accuracy.

### Introduction
Effective training of Deep Neural Networks (DNNs) often requires access to vast amounts of data, posing challenges in privacy and communication costs. Federated Learning (FL) algorithms have emerged as a solution, but face challenges in heterogeneous environments. ComDML introduces a decentralized approach, balancing workload to minimize training time. Traditional FL methods rely on a central server, prone to bottlenecks and failures. Decentralized systems offer improved resilience and security. ComDML addresses workload balancing challenges without a central coordinator.

### Workload Balancing for Decentralized Multi-agent Learning
ComDML employs local-loss-based split training to achieve workload balancing in DML systems. This approach allows slower agents to offload part of their workload to faster agents, ensuring efficient utilization of resources. The optimization for workload balancing considers both communication and computation resources, formulated as an integer programming problem. A dynamic decentralized pairing scheduler efficiently pairs agents based on observed capabilities.

### Decentralized Workload Balancing
To effectively implement workload balancing, ComDML utilizes a dynamic decentralized pairing scheduler. This scheduler pairs agents based on their computation and communication capacities, minimizing overall training time. The training workflow involves split model profiling, agent pairing, and model aggregation, achieving resource optimization with minimal overhead.


