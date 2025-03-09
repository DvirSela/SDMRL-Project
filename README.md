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
Each following sections should be run in the order we describe. The running is splitted into 2 parts - training and evaluation. To change any hyper-parameters, just edit the provide [.env](helpers%20.env) file. 

## Training
This is the main part of the code, which will train the SAC and TAC agents. The releavant Hyper-Parameters in the `.env` are:
```python

TRAIN_TIME_STEPS - The number of training steps the models will do each episode.
```
The parameters are:
- `N` - number of samples
- `seed` - seed for randmoness
- `k` - number of users to get for each job listing
- `show_null` - bool, will determine if to show null checks or skip
- `save_to_dbfs` - bool, will determine if to save results to dbfs
- `index_model` - the index of the chosen model in the model list
- `models_list` - list of models to use for the embeddings
## Evaluating
This is the part of the code that runs the evaluations and visualiztions.
- The notebook for plotting the T-SNE is [TSNE.ipynb](Databricks%20Code/TSNE.ipynb)
