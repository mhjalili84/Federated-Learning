#Federated Learning Models

#This repository contains a collection of Federated Learning (FL) implementations developed to deepen understanding of the methodology, explore its challenges, and identify opportunities for research and real-world applications.

#Overview
#Federated Learning is a decentralized machine learning approach where multiple clients collaboratively train a model without sharing raw data. This paradigm is essential for applications where data privacy, communication efficiency, and heterogeneous environments are major concerns.
In this project, I have implemented and experimented with multiple FL strategies, including:
-FedAvg (Federated Averaging)
-FedBuff (Buffered Asynchronous Federated Learning)
Each implementation is designed to help analyze:
-Client communication delays
-Impact of asynchronous updates
-Behavior under non-IID data distribution
-Buffer management strategies
-Model convergence and accuracy in distributed scenarios

Features
-Modular and extensible codebase using PyTorch
-Simulation of per-client communication delays (mean & std definable)
-Dynamic buffer-based aggregation (FedBuff-style)
-Support for non-IID data partitioning (Dirichlet distribution)
-Visualization of training performance (loss & accuracy)
