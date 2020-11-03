# CoDA CoDE Supplement

This contains an implementation of CoDA and the Batch RL / Multi-goal experiments used in our NeurIPS 2020 paper *Counterfactual Data Augmentation using Locally Factored Dynamics*. [Arxiv link](https://arxiv.org/abs/2007.02863). 


## Installation

There is a `requirements.txt` that was works with venv:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then install the appropriate version of `Pytorch` by following the instructions here: https://pytorch.org/get-started/locally/.

To run `Mujoco` environments you need to have the Mujoco binaries and a license key. Follow the instructions [here](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key).

To test that everything compiles run:

```
PYTHONPATH=./ pytest tests
```

## Relevant pieces of code

- For implementation of `CoDA` (Algorithm 1) see `coda_generic.py`.
- For how `CoDA` (Algorithm 1) is used, see `coda_module.py`.
- For implementation of the transformer masking model (`SANDy-Transformer` in Appendix) see `sandy_module.py`.

## Running Experiments

#### Pong

From the main folder, run the experiment:

```
PYTHONPATH=./ python experiments/coda/pong/pong_experiment.py --seed 0 --num_real_samples 25000 --num_coda_samples 25000
```

(Should replicate corresponding results in paper (25K real data size with 1:1 Real:CoDA ratio))

#### Fetch

From the main folder, run the experiment:

```
PYTHONPATH=./ python experiments/coda/train_coda.py --env disentangledpush --tb CODA --replay_size 1000000 --coda_buffer_size 3000000 --batch_size 2000 --her futureactual_2_2 --max_steps 1000000 --coda_every 250 --coda_source_pairs 2000 --relabel_type push_heuristic --max_coda_ratio 0.75 --seed 111 --parent_folder ./push_results --num_envs 6
```

(Should achieve test reward better than -40 within 30,000 steps and better than -25 in 50,000 steps)
