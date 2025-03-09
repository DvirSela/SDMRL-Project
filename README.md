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
- [Overview](#Overview)
- [Abstract](#Abstract)
- [Running the code](#Running-the-code)
  - [Scraping](#Scraping) 🪥
  - [Assign workers to job listings](#Assign-Workers-to-Job-Listing)👜
  - [Running tests and visualiztions](#Running-tests-and-visualiztions) 🔬
  
# Overview

Our goal is to develop a recommendation system that helps companies and employers connect with job candidates who are the best fit for the job.

# Abstract

In this project, we test various reinforcement learning (RL) techniques on a simulated environment of a electricity market. We begin by describing the properties and dynamics of our simulation environment, and then we explain our RL methods, such as Soft Actor-Critic (SAC) and our novel Transformer Actor-Critic (TAC). Overall, we find that both the SAC and TAC perform about as well as one another, but relatively well overall relative to a random baseline and also relative to a heuristic-based baseline.

# Running the code
Each following sections should be run in the order we describe. Farthermore, after scraping we uploaded the scraped data in to the given DataBricks server, and it should be taken into account when running localy. 
##  Scraping
The code for scraping [monster.com](https://www.monster.com/) jobs is found in [scraping.ipynb](Scraping%20Code/scraping.ipynb).<br>
You should provide a .env file with the following parameters:
```python
USER = 'USER'  # BrightData username
PASS = 'PASS'  # BrightData password
```
## Assign Workers to Job Listings
This is the main part of the code, that matches job listing into K users. The code is found in [main.ipynb](Databricks%20Code/main.ipynb).<br> 
You will be greeted with the following parameters, and can change them at your will to get different results:
```python
N = 20_000
seed = 42
k = 10
max_sentence_length = 512
show_null = True
save_to_dbfs = True
index_model = 1
models_list = ['all-MiniLM-L6-v2', # Bert
                'all-distilroberta-v1', # roberta
                'multi-qa-distilbert-cos-v1' # distilberta
                ]
```
The parameters are:
- `N` - number of samples
- `seed` - seed for randmoness
- `k` - number of users to get for each job listing
- `show_null` - bool, will determine if to show null checks or skip
- `save_to_dbfs` - bool, will determine if to save results to dbfs
- `index_model` - the index of the chosen model in the model list
- `models_list` - list of models to use for the embeddings
## Running tests and visualiztions
This is the part of the code that runs the tests and visualiztions.
- The notebook for plotting the T-SNE is [TSNE.ipynb](Databricks%20Code/TSNE.ipynb)
- The notebook for plotting meta-industry percentages is [meta_dist.ipynb](Databricks%20Code/meta_dist.ipynb)
- The notebook for running the statistical test as mentioned in the report is [statistical_test.ipynb](Databricks%20Code/statistical_test.ipynb)
