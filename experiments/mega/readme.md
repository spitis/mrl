# Maximum Entropy Gain Exploration for Long Horizon Multi-goal Reinforcement Learning

### [Silviu Pitis](https://silviupitis.com)\*, [Harris Chan](https://takonan.github.io/)\*, Stephen Zhao, Bradly Stadie, Jimmy Ba

This is code for replicating our [ICML 2020 paper](https://arxiv.org/abs/2007.02832). See also the [ALA 2020 presentation (25 minutes, best paper)](https://underline.io/lecture/544-maximum-entropy-gain-exploration-for-long-horizon-multi-goal-reinforcement-learning).

See the `mrl` [readme](https://github.com/spitis/mrl) for general instructions. 

The launch commands used for the main experiments are in `commands_for_experiments.txt`, which call the `train_mega.py` launch script. 

To run a MEGA agent use `--ag_curiosity minkde`. To use OMEGA use `--transition_to_dg`. The actual implementation of MEGA is the `DensityAchievedGoalCuriosity` from `mrl.modules.curiosity`, which assumes the agent has a density module from `mrl.modules.density` (KDE / Flow / RND).

While the paper experiments use the `protoge_config` from `mrl.configs.continuous_off_policy.py`, please note that the `best_slide_config` hparams with `--optimize_every 10 --her rfaab_1_5_2_1_1` are much more stable for Stack (and likely FPP, etc.).


### Bibtex

```
@inproceedings{pitis2020mega,
  title={Maximum Entropy Gain Exploration for Long Horizon Multi-goal Reinforcement Learning},
  author={Pitis, Silviu and Chan, Harris and Zhao, Stephen and Stadie, Bradly and Ba, Jimmy},
  booktitle={Proceedings of the Thirty-seventh International Conference on Machine Learning},
  year={2020}
}
```
