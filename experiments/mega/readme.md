# MEGA

This is code for replicating our MEGA paper. 

The launch commands used for the main experiments are in `commands_for_experiments.txt`, which call the `train_mega.py` launch script. 

To run a MEGA agent use `--ag_curiosity minkde`. To use OMEGA use `--transition_to_dg`. The actual implementation of MEGA is the `DensityAchievedGoalCuriosity` from `mrl.modules.curiosity`, which assumes the agent has a density module from `mrl.modules.density` (KDE / Flow / RND).

While the paper experiments use the `protoge_config` from `mrl.configs.continuous_off_policy.py`, please note that the `best_slide_config` hparams with `--optimize_every 10 --her rfaab_1_5_2_1_1` are much more stable for Stack (and likely FPP, etc.).