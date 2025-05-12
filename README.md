# Federated Learning Models

A collection of Federated Learning (FL) implementations developed to explore the core concepts, challenges, and opportunities of decentralized machine learning.

## Table of Contents

- [Overview](#overview)  
- [Implemented Algorithms](#implemented-algorithms)  
- [Key Features](#key-features)  
- [Getting Started](#getting-started)  
- [Usage Example](#usage-example)  
- [Motivation](#motivation)  
- [License](#license)

---

## Overview

Federated Learning allows multiple clients to collaboratively train a machine learning model without sharing their local data. This approach enhances data privacy and is especially useful in real-world applications involving sensitive information and distributed sources.

This repository serves as a sandbox for experimenting with FL approaches and understanding their performance in asynchronous and non-IID environments.

---

## Implemented Algorithms

- **FedAvg** – Federated Averaging  
- **FedBuff** – Buffered Asynchronous Federated Learning  
- **FD** - Federated Distillation

---

## Key Features

- Simulates **asynchronous updates** with per-client communication delays  
- Allows **customizable delay distributions** (mean & std for each client)  
- Non-IID data partitioning using **Dirichlet distribution**  
- Modular structure built with PyTorch  
- **Buffer-based aggregation** strategies  
- Accuracy & loss tracking with visualization  

---

## Getting Started

### Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib

