from mrl.configs.make_continuous_agents import *
import numpy as np
import pytest

def test_sac():
  config = make_sac_agent(args=Namespace(env='InvertedPendulum-v2',
                                    tb='',
                                    parent_folder='/tmp/mrl',
                                    layers=(32, 1),
                                    num_envs=1,
                                    device='cpu'))
  agent = mrl.config_to_agent(config)

  agent.train(num_steps=1)
  assert len(agent.eval(num_episodes=1).rewards) == 1