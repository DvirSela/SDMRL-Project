<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> SDMRL-Project</h1>

<p align='center' style="text-align:center;font-size:1em;">
    Dvir Sela
    <br/> 
    Peleg Michael
    <br/> 
    Technion - Israel Institute of Technology
</p>

<br>
<br>

# Contents
- [Abstract](#Abstract)
- [Running the code](#Running-the-code)
  - [Training](#Training) 🚂
  - [Evaluating](#Evaluating) 🔬

# Abstract

In this project, we test various reinforcement learning (RL) techniques on a simulated environment of a electricity market. We begin by describing the properties and dynamics of our simulation environment, and then we explain our RL methods, such as Soft Actor-Critic (SAC) and our novel Transformer Actor-Critic (TAC). Overall, we find that both the SAC and TAC perform about as well as one another, but relatively well overall relative to a random baseline and also relative to a heuristic-based baseline.

# Running the code
Each following sections should be run in the order we describe. The running is slitted into 2 parts - training and evaluation. To change any hyper-parameters, just edit the provide [.env](.env) file. 

## Training
This is the main part of the code, which will train the SAC and TAC agents. The relevant Hyper-Parameters in the `.env` are:
- `TRAIN_TIME_STEPS` - int, number of training steps the models will perform in each episode.  
- `NUM_EPISODES` - int, number of episodes to run during training.  
- `TRAIN` - bool, determines whether to overwrite existing training data or just load from previous runs.  
- `INITIAL_SOC` - float, initial state of charge for the battery.  
- `BATTERY_CAPACITY` - float, defines the total battery capacity.  
- `RENEWABLE_SCALE` - float, controls the scale at which the agent receives renewable energy; higher values mean more energy.  
- `DEMAND_NOT_MET_FACTOR` - float, penalty factor for unmet demand; higher values result in lower punishment.  
- `HISTORY_LENGTH` - int, only relevant for TAC, defines the length of saved history.  
- `TAC_BATCH_SIZE` - int, only relevant for TAC, determines the batch size used in training.  
- `REPLAY_BUFFER_CAPACITY` - int, only relevant for TAC, sets the capacity of the replay buffer.  

In order to train the models, just run [train_SAC.py](train_SAC.py) for the SAC and [train_TAC.py](train_TAC.py) for the TAC.
## Evaluating
This is the part of the code that runs the evaluations and visualizations. The relevant Hyper-Parameters in the `.env` and are the same as before. 
All the needed visualizations and evaluations that were in our report + extra are located in the [evaluations.py](evaluations.ipynb) notebook. Make sure you run the training before you submit.
