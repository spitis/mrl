# `mrl`: modular RL

This is a modular RL code base for research. The intent is to enable surgical modifications by designing the base agent as a list of modules that all live inside the agent's global namespace (so they can all access each other directly by name). This means we can change the algorithm of a complex hierarchical, multi-goal, intrinsically motivated, etc. agent from DDPG to SAC by simply changing the algorithm module (and adding the additional critic network). Similarly, to add something like a forward model, intrinsic motivation, landmark generation, a new HER strategy, etc., you only need to create/modify the relevant module(s). 

The agent has life-cycle hooks that the modules "hook" into. The important ones are: `_setup` (called after all modules are set but before any environment interactions), `_process_experience` (called with each new experience), `_optimize` (called at each optimization step), `save/load` (called upon saving / loading the agent). 

See comments in `mrl/agent_base.py`, brief test scripts in `tests`, and example TD3/SAC Mujoco agents in `experiments/benchmarks/train_online_agent.py`. 

The modular structure is technically framework agnostic, so could be used with either Pytorch or TF-based modules, or even a mix, but right now all modules that need a framework use Pytorch. 

Train loop is easily customized, so that you can do, e.g., BatchRL, transfer, or meta RL with minimal modifications. 

Environment parallelization is done via VecEnv, and we rely on GPU for optimization parallelization. Future work should consider how they can be done asynchronously; e.g., using [Ray](https://ray.readthedocs.io/en/latest/). 


### Performance Benchmarks

mrl provides state of the art implementations of SAC, TD3, and DDPG+HER. See the [Mujoco and Multi-goal benchmarks](https://github.com/spitis/mrl/blob/master/experiments/benchmarks/readme.md). 


## Installation

There is a `requirements.txt` that was works with venv:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then `pip install` the appropriate version of `Pytorch` by following the instructions here: https://pytorch.org/get-started/locally/.

To run `Mujoco` environments you need to have the Mujoco binaries and a license key. Follow the instructions [here](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key).

To test run:

```
pytest tests
PYTHONPATH=./ python experiments/mega/train_mega.py --env FetchReach-v1 --layers 256 256 --max_steps 5000
```

The first command should have 3/3 success.
The second command should solve the environment in <1 minute (better than -5 avg test reward). 

## Usage

To understand how the code works, read `mrl/agent_base.py`. 

See `tests/test_agent_sac.py` and `experiments/benchmarks` for example usage. The basic outline is as follows:

1. Construct a config object that contains all the agent hyperparameters and modules. There are some existing base configs / convenience methods for creating default SAC/TD3/DDPG agents (see, e.g., the benchmarks code). If you use `argparse` you can use a config object automatically populate the parser using `parser = add_config_args(parser, config)`. 
2. Call `mrl.config_to_agent` on the config to get back an agent. 
3. Use the agent however you want; e.g., call its train/eval methods, save/load, module methods, and so on. 

To add functionality or a new algorithm, you generally just need to define a one or more modules that hook into the agent's lifecycle methods and add them to the config. They automatically hook into the agent's lifecycle methods, so the rest of the code can stay the same. 

## Implemented / Outstanding

Implemented:
- [DDPG](https://arxiv.org/abs/1509.02971), [TD3](https://arxiv.org/abs/1802.09477), [SAC](https://arxiv.org/abs/1801.01290), [basic DQN](https://arxiv.org/abs/1312.5602)
- [HER](https://arxiv.org/abs/1707.01495) (computed online) 
- Random ensemble DDPG (based on [An Optimistic Perspective on Offline Reinforcement Learning](https://arxiv.org/abs/1907.04543) --- could be improved)
- N-step returns (computed online) (see [Rainbow](https://arxiv.org/pdf/1710.02298.pdf)) [not compatible with HER]
- MLE versions of DDPG/TD3 using Gaussian critic (called ``Sigma'' in the code, cf. [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474))
- Fixed-horizon DDPG (based on [Fixed-Horizon Temporal Difference Methods](https://arxiv.org/abs/1909.03906))
- Gamma as an auxiliary task (just pass a vector of k gammas and use a k-output critic --- last Gamma will be used to train policy) (based on [Hyperbolic Discounting and Learning over Multiple Horizons](https://arxiv.org/abs/1902.06865))
- Support for goal-based intrinsic motivation, in goal-based environments 

Some todos:
- Distributional predictions
- Uncertainty predictions
- Hierarchical RL
- Support for goal-based intrinsic motivation in general environments
- DQN variants

## Papers using this Repository

Below is a list of papers that use `mrl`. If you use `mrl` in one of your papers please let us know and we can add you to the list. If you build on the experiments related to the below papers, please cite the original papers:

- Maximum Entropy Gain Exploration for Long Horizon Multi-goal Reinforcement Learning (ICML 2020 ([15 minute presentation](https://icml.cc/virtual/2020/poster/6622)), [Arxiv](https://arxiv.org/abs/2007.02832), ALA 2020 Best Paper ([25 minute presentation](https://bit.ly/mega_ala))
- ProtoGE: Prototype Goal Encodings for Multi-goal Reinforcement Learning (RLDM 2019, [pdf](https://takonan.github.io/docs/2019_protoge_rldm.pdf)) [As of July 2020, this is still far and away the state-of-the-art on Gym's Fetch environments]
- Counterfactual Data Augmentation using Locally Factored Dynamics (Preprint, [Arxiv](http://arxiv.org/abs/2007.02832
))


## Citing this Repository

If you use or extend this codebase in your work, please consider citing:

```
@misc{mrl,
  author = {Pitis, Silviu and Chan, Harris and Zhao, Stephen},
  title = {mrl: modular RL},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/spitis/mrl}},
}
```

## References

This code has used parts of the following repositories:

- [DeepRL by Shangtong Zhang](https://github.com/ShangtongZhang/DeepRL) (parts of networks, normalizer, random processes / schedule)
- [Spinning Up](https://spinningup.openai.com/en/latest/index.html) (parts of logger, certain code optimizations)
- [Baselines](https://github.com/openai/baselines) (VecEnv, RingBuffer, normalizer, plotting)

## Contributors

[Silviu Pitis](https://silviupitis.com) ([spitis](https://github.com/spitis)), [Harris Chan](https://takonan.github.io/) ([takonan](https://github.com/Takonan)), Stephen Zhao ([Silent-Zebra](https://github.com/Silent-Zebra))