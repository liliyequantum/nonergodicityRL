# Nonergodicity_RL_control for 1D tilted Fermi-Hubbard chain

## System requirement

Windows 10, CPU or GPU for 14 number of lattice

## Setup and Running the Code

To set up the environment, open the Anaconda Prompt and run the following commands:

```sh
conda create -n qmbs python=3.9 #if you need to delete env, use this command 'conda remove --name qmbs --all'
conda env list
conda activate qmbs
pip install -r requirements.txt
pip install spyder
spyder #Open spyder by running the following command in the Anaconda Prompt
```

In Matlab, please install Reinforcement Learning Toolbox.


## Code Files and Running order

1. In directory `time_evolu_sub_chain`\\
  a. `basisAndHamiltonian.py` is used to construct the Hamiltonian matrix and its basis with output `data.mat`;
  b. `pre_data.m` is used to process data from `data.mat` with output `pre_data.mat`
  c. `time_evo.m` is used to do the numerical simulation, save, and plot resulting fidelity, imbalance, and entropy.

2. In remaining directories `RL_obser_imbalance`, `RL_obser_F_sub`, and `RL_obser_F_full`\\
   The same runing order a. and b. Next, run `PPO.m` and then `test_PPO.m`
   
## Tips

First run the code, suggest set `numLattice = 8 or 10`, `n_steps = 10`,  `n_episode=10`, `output_interval = 1` 
