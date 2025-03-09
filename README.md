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
Each following sections should be run in the order we describe. The running is splitted into 2 parts - training and evaluation. To change any hyper-parameters, just edit the provide [.env](.env) file. 

## Training
This is the main part of the code, which will train the SAC and TAC agents. The releavant Hyper-Parameters in the `.env` are:
```python
TRAIN_TIME_STEPS - int. The number of training steps the models will do each episode.
NUM_EPISODES - int. number of episodes to rnu the training.
TRAIN - bool. decides if to overwrite or not when training (if not, will just load)
INITIAL_SOC - float. inital state_of_charge
BATTERY_CAPACITY - float. decides the battery_capacity
RENEWABLE_SCALE - flaot. decides the scale at which the agent get renewable enerygt. higher is more.
DEMAND_NOT_MET_FACTOR - float. how much to punish when not meeting the demand. Higher factor means lower punishment.
HISTORY_LENGTH - int. Only relevant for the TAC. Decides the length the saved history
TAC_BATCH_SIZE - int. Only relevant for the TAC. Decides the batch size
REPLAY_BUFFER_CAPACITY - int. Only relevant for the TAC. Decides the replay buffer capacity.
```
In order to train the models, just run [train_SAC.py](train_SAC.py) for the SAC and [train_TAC.py](train_TAC.py) for the TAC.
## Evaluating
This is the part of the code that runs the evaluations and visualiztions.
- The notebook for plotting the T-SNE is [TSNE.ipynb](Databricks%20Code/TSNE.ipynb)
