from experiments.coda.pong.collect_real_data import collect_real_data
from experiments.coda.pong.make_coda_data import make_coda_data
from experiments.coda.pong.make_dyna_data import make_dyna_data
from experiments.coda.pong.make_mbpo_data import make_dyna_data as make_mbpo_data
from experiments.coda.pong.train_batchrl_agent import train_batchrl_agent
import numpy as np

def run_pong_experiment(args):
  dataset = collect_real_data(args.num_real_samples, seed=args.seed, noise_level=args.noise_level)
  reward_fn = None
  amt_real_data = None

  if args.num_mbpo_samples > 0:

    if args.use_3x_coda_to_train_mbpo:
      # First expand the dataset using CoDA so can train a better MBPO model
      dataset = make_coda_data(dataset, 3*args.num_real_samples, seed=args.seed+123)

    dataset, reward_fn = make_mbpo_data(dataset, args.num_mbpo_samples, args.num_step_rollouts, seed=args.seed)

    if args.use_3x_coda_to_train_mbpo:
      # But take out the original CoDA data---we will remake it after the MBPO data is made
      dataset = [np.concatenate((x[:args.num_real_samples], x[4*args.num_real_samples:4*args.num_real_samples+args.num_mbpo_samples])) for x in dataset]

    amt_real_data = args.num_real_samples

  if args.num_dyna_samples > 0:
    assert args.num_coda_samples == 0
    assert args.num_mbpo_samples == 0
    dataset = make_dyna_data(dataset, args.num_dyna_samples, args.num_step_rollouts, seed=args.seed)
  else:
    dataset = make_coda_data(dataset, args.num_coda_samples, seed=args.seed, amt_real_data=amt_real_data, reward_fn=reward_fn)

  mbpo = args.num_mbpo_samples if not args.use_3x_coda_to_train_mbpo else 0
  c3xm = args.num_mbpo_samples if args.use_3x_coda_to_train_mbpo else 0

  train_batchrl_agent(dataset, f'batchrl_real{args.num_real_samples}_coda{args.num_coda_samples}_dyna{args.num_dyna_samples}_mbpo{mbpo}_c3xm{c3xm}_roll{args.num_step_rollouts}_noise{args.noise_level}_{args.tag}', num_steps=args.num_steps, seed=args.seed, results_folder=args.parent_folder)

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(
      description="Run BatchRL on Pong with Coda",
      formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='/tmp/pong_results', type=str, help='Where to save the batchRL agent (note: data is not saved)')
  parser.add_argument('--tag', default='', type=str, help='Additional tag to name the agent.')
  parser.add_argument('--num_steps', type=int, default=500000, help='Number of train steps for the RL agent')
  parser.add_argument('--noise_level', type=float, default=0.5, help='How often expert takes a random action.')

  # Required args
  parser.add_argument('--seed', type=int, required=True)
  parser.add_argument('--num_real_samples', type=int, required=True)
  parser.add_argument('--num_coda_samples', type=int, required=True)

  # Also Dyna (cannot have both CODA + DYNA, BUT can have both CODA + MBPO)
  parser.add_argument('--num_dyna_samples', type=int, default=0)
  parser.add_argument('--num_mbpo_samples', type=int, default=0)
  parser.add_argument('--num_step_rollouts', type=int, default=5)
  parser.add_argument('--use_3x_coda_to_train_mbpo', action='store_true')

  args = parser.parse_args()
  run_pong_experiment(args)