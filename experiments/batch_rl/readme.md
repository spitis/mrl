# Batch Rl stuff

This is just a boilerplate for running some Batch RL experiments. Of course, the offline SAC that is included here fails (as is expected). You will need a special algorithm (not included) like [CQL](https://arxiv.org/abs/2006.04779) [might be coming soon!]. 

Can install [d4rl](https://github.com/rail-berkeley/d4rl) for access to their datasets. 

Can also prepare own datasets using scripts `collect_policies` and `policy_to_buffer`, as follows:

    PYTHONPATH=./ python experiments/batch_rl/collect_policies.py --parent_folder /home/silviu/Documents/batchrl/ --env Ant-v3 --num_envs 4
    PYTHONPATH=./ python experiments/batch_rl/collect_policies.py --parent_folder /home/silviu/Documents/batchrl/ --env HalfCheetah-v3 --num_envs 4
    PYTHONPATH=./ python experiments/batch_rl/collect_policies.py --parent_folder /home/silviu/Documents/batchrl/ --env Hopper-v3 --num_envs 4
    PYTHONPATH=./ python experiments/batch_rl/collect_policies.py --parent_folder /home/silviu/Documents/batchrl/ --env Walker2d-v3 --num_envs 4

    PYTHONPATH=./ python experiments/batch_rl/policy_to_buffer.py --parent_folder /home/silviu/Documents/batchrl/ --env Ant-v3 --num_envs 10 --load_folder /home/silviu/Documents/batchrl/batchrl_env-Ant-v3_alg-sac_seed0_tb-2/performance_6903/
    PYTHONPATH=./ python experiments/batch_rl/policy_to_buffer.py --parent_folder /home/silviu/Documents/batchrl/ --env HalfCheetah-v3 --num_envs 10 --load_folder /home/silviu/Documents/batchrl/batchrl_env-HalfCheetah-v3_alg-sac_seed0_tb-2/performance_16669/
    PYTHONPATH=./ python experiments/batch_rl/policy_to_buffer.py --parent_folder /home/silviu/Documents/batchrl/ --env Hopper-v3 --num_envs 10 --load_folder /home/silviu/Documents/batchrl/batchrl_env-Hopper-v3_alg-sac_seed0_tb-2/performance_4191/
    PYTHONPATH=./ python experiments/batch_rl/policy_to_buffer.py --parent_folder /home/silviu/Documents/batchrl/ --env Walker2d-v3 --num_envs 10 --load_folder /home/silviu/Documents/batchrl/batchrl_env-Walker2d-v3_alg-sac_seed0_tb-2/performance_6612/


Then we want to train an agent with `train_batchrl.py`... 
