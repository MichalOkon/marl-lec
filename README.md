# marl-lec
The codebase for my master's thesis: "Uncovering Sequential Social Dilemmas in Multi-Agent Reinforcement Learning: Challenges and Strategies for Local Energy
Communities".

In this repository, you will find the codebase for the environment presented in the thesis along with the conducted experiments.

In order to run the code the data coming from the Pecan Street dataset needs to be downloaded from the following link: https://www.pecanstreet.org/dataport/.
The full 15-minute resolution New York dataset is used in the experiments. The metadata along with the timeseries data should be placed in the `data/pecan_street` folder
and should be named `metadata.csv` and `15minute_data_newyork.csv` respectively.

The code required to run the specific experiments is located in notebooks inside the root directory.

The exact packages used to run this codebase can be found in the `environment.yml` file.