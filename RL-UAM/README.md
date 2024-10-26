# Online Optimization Framework for Urban Air Mobility

## Overview
This project aims to utilize Reinforcement Learning (RL) to optimize strategies within the VertiSim simulation environment. VertiSim models a vertical takeoff and landing (VTOL) air traffic environment where the RL agent will devise control strategies for aircraft to optimize various objectives like minimizing travel time, reducing energy consumption, optimizing dispatch and charge policies and more.

## Key Components

- **VertiSim**: A simulation environment for vertiport network, offering an API for interaction and obtaining observations/states.
  
- **RL Algorithm**: A model that interacts with VertiSim through API calls to train control strategies and policies for the agents (aircraft) in the simulation.

- **Service Orchestrator**: Manages the instances of VertiSim, handling creation, reset, and termination.

- **Database Service**: Manages and stores simulation and training data.

## Installation and Setup

### Prerequisites

- Python 3.11+
- Docker
- Some knowledge in Reinforcement Learning and Simulations is beneficial for understanding and potentially extending the project.

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone [repository_url] rl_uam
   cd rl_uam
