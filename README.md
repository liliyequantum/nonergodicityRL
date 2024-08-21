# nonergodicity_RL_control for 1D tilted Fermi-Hubbard chain

## System requirement

Windows 10, CPU or GPU for 14 number of lattice

## Setup

To set up the environment, open the Anaconda Prompt and run the following commands:

```sh
conda create -n qmbs python=3.9
conda env list
conda activate qmbs
pip install spyder
pip install -r requirements.txt
```

## File Placement for the Custom Quantum Environment
1. Put all files from the directory `.\cust_env\classical_control\` into the `qutip_RL` environment directory at `C:\users\yourUserName\anaconda3\envs\qutip_RL\Lib\site-packages\gym\envs\classic_control\`, replacing the original files.

2. Copy the file from `.\cust_env\__init__.py` and paste it into `C:\users\yourUserName\anaconda3\envs\qutip_RL\Lib\site-packages\gym\envs\__init__.py`, replacing the original file.

## Running the Code

Open spyder by running the following command in the Anaconda Prompt
```sh
spyder
```

## Code Files

1. Files with the suffix `fig_plot` are used for plotting figures.

2. Files with the suffix `fig_code` are used for plotting partial figures.

3. Files with the suffix `fig_data` are used for generating data for figures.

## Tips

1. When running the training code (e.g., `training_fig2_data.py`), you can copy it into a new directory named code_test. In this directory, you can try reducing the training load by setting parameters such as `n_episode = 10`, `n_steps = 10`, `n_update = 2`, and `output_interval = 2`. This will allow you to quickly test the code.

2. The testing process in the code (e.g., `test_ave_fig2_code.py`) has the testing function (e.g., `PPOtest`) commented out and saves all testing data. You can simply plot all results by running such codes (e.g., `test_ave_fig2_code.py`) directly.

## Data
The complete training results from Figures 4 to 8 and Figures S1 to S2 have been shared on Zenodo: [https://doi.org/10.5281/zenodo.12584159]

## References

1. Stable-baselines3 for the PPO agent: [https://stable-baselines3.readthedocs.io/en/master/index.html]

2. Sb3-contrib for the recurrent PPO agent: [https://sb3-contrib.readthedocs.io/en/master/index.html]

3. QuTip: [https://qutip.readthedocs.io/en/master/index.html]
