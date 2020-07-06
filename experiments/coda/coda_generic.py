"""
Generic rewrite of CODA for set-based state representations. 
"""
import numpy as np
from functools import reduce
from itertools import combinations, chain
from scipy.sparse.csgraph import connected_components
from mrl.utils.misc import batch_block_diag, batch_block_diag_many
import random
import copy
import multiprocessing as mp

FETCH_HEURISTIC_IDXS = [[0,1,2], [10,11,12], [22,23,24], [34,35,36], [46,47,47], [58,59,60], [70,71,72]]

def get_true_abstract_mask_spriteworld(sprites, config, action=(0.5, 0.5)):
  """Returns a mask with iteractions for next transition given true sprites.
  E.g., returns [[1,0,0],[0,1,0],[0,0,1]] for 3 sprites"""
  sprites1 = copy.deepcopy(sprites)
  config['action_space'].step(action, sprites1)
  return config['renderers']['mask_abstract'].render(sprites1)

def batch_get_heuristic_mask_fetchpush_(s, a):
  """ heuristic mask for disentangled fetch """
  grip_poses = s[:, 0, :3]
  obj_poses  = s[:, 1, 10:13]

  entangled = np.linalg.norm(grip_poses - obj_poses, axis=1, keepdims=True)[:, None] < 0.1
  
  entangled_mask = np.ones((1, 3, 3))
  disentangled_mask = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]]])

  return np.where(entangled, entangled_mask, disentangled_mask)

def batch_get_heuristic_mask_fetchpush(s, a):
  poses = np.stack([s[:, i, h] for i, h in enumerate(FETCH_HEURISTIC_IDXS[:s.shape[1]])], 1) # batch x n_state_comps x 3
  dists = np.linalg.norm(poses[:,None] - poses[:,:,None], axis = -1) # batch x n_state_comps x n_state_comps
  entangled = (dists < 0.1).astype(np.float32)
  entangled = np.pad(entangled, ((0, 0), (0, 1), (0, 1))) # add action dim
  entangled[:,-1, 0] = 1. # entangle action with gripper
  
  return entangled


def get_default_mask(s, a):
  """ assumes set-based s and a... so shape should be (n_componenents, *component_shape) """
  if len(a.shape) == 1:
    a.reshape(1, -1)
  
  mask_dim = len(s) + len(a)
  mask_shape = (mask_dim, mask_dim)
  return np.ones(mask_shape)

def batch_get_default_mask(s, a):
  """ Batch version of get default mask """
  s_shape = s.shape
  a_shape = a.shape
  if len(a_shape) == 2:
    assert len(a_shape) > 1
    a.reshape(-1, 1, a_shape[-1])
  
  mask_dim = s_shape[1] + a_shape[1]
  mask_shape = (s_shape[0], mask_dim, mask_dim)
  return np.ones(mask_shape)




def get_cc_from_mask(mask):
  """
  Converts a mask into a list of CC indices tuples.
  E.g., if mask is [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]],
  this will return [array([0]), array([1]), array([2, 3])]
  
  Note that the mask should be a square, so in case we have (s, a) x (s2,),
  we should first dummy a2 columns to form a square mask. 
  """
  ccs = connected_components(mask)
  num_ccs, cc_idxs = ccs
  return [np.where(cc_idxs == i)[0] for i in range(num_ccs)]

def powerset(n):
  xs = list(range(n))
  return list(chain.from_iterable(combinations(xs, n) for n in range(n + 1)))

def reduce_cc_list_by_union(cc_list, max_ccs):
  """Takes a cc list that is too long and merges some components to bring it
  to max_ccs"""
  while len(cc_list) > max_ccs:
    i, j = np.random.choice(range(1, len(cc_list) - 1), 2, replace=False)
    if (j == 0) or (j == len(cc_list) - 1):
      continue  # don't want to delete the base
    cc_list[i] = np.union1d(cc_list[i], cc_list[j])
    del cc_list[j]
  return cc_list

def disentangled_components(cc_lst):
  """Converts connected component list into a list of disentangled subsets
  of the indices.
  """
  subsets = powerset(len(cc_lst))
  res = []
  for subset in subsets:
    res.append(reduce(np.union1d, [cc_lst[i] for i in subset], np.array([])).astype(np.int64))
  return set(map(tuple, res))

def get_dcs_from_mask(mask, max_ccs = 6):
  cc = get_cc_from_mask(mask)
  return disentangled_components(reduce_cc_list_by_union(cc, max_ccs))

def transitions_and_masks_to_proposals(t1,
                                       t2,
                                       m1,
                                       m2,
                                       max_samples=10,
                                       max_ccs=6):
  """ 
  assumes set-based s and a... so shape should be (n_components, *component_shape)
  Takes two transitions with their masks, and combines them
  using connected-component relabeling to form proposals

  Returns a list of tuples of ((s1, a1, s2) proposal, disconnected_component_idxs).
  """
  sa1, s21 = t1
  sa2, s22 = t2

  # get_dcs_from_mask should return a set of tuples of indices, inc. the empty tuple
  # where the subgraph represented by each tuple is disconnected from the result of 
  # the graph. Note that mask should be square, so columns corresp. to action idxs are 
  # dummy columns.
  #
  # E.g., if mask is [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]],
  # this function should return:
  #   set([ (,), (0,), (1,), (0,1), (2, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)  ])

  dc1 = get_dcs_from_mask(m1, max_ccs)
  dc2 = get_dcs_from_mask(m2, max_ccs)

  # get shared connected components in random order
  shared_dc = list(dc1.intersection(dc2))
  random.shuffle(shared_dc)

  # subsample shared_dc down to max_samples
  if len(shared_dc) > max_samples:
    shared_dc = shared_dc[:max_samples]

  all_idxs = set(range(len(sa1)))

  res = []
  for dc in shared_dc:
    not_dc = list(all_idxs - set(dc))
    dc = list(dc) # (0, 2)

    proposed_sa = np.zeros_like(sa1)
    proposed_s2 = np.zeros_like(sa1)

    proposed_sa[dc]     = sa1[dc]
    proposed_sa[not_dc] = sa2[not_dc]
    proposed_s2[dc]     = s21[dc]
    proposed_s2[not_dc] = s22[not_dc]

    proposed_t = (proposed_sa, proposed_s2)
    res.append((proposed_t, tuple(dc)))

  return res


def relabel_independent_transitions_spriteworld(t1,
                                    sprites1,
                                    t2,
                                    sprites2,
                                    config,
                                    reward_fn=lambda _: 0,
                                    total_samples=10,
                                    flattened=True,
                                    custom_get_mask=None,
                                    max_ccs=6):
  """
  Same as old fn, but using new refactored bits -- to make sure it works
  """
  assert flattened

  s1_1, a_1, _, s2_1 = t1
  s1_2, a_2, _, s2_2 = t2

  num_sprites = len(sprites1)
  num_feats = len(s1_1) // num_sprites

  t1 = (batch_block_diag(s1_1.reshape(num_sprites, num_feats), a_1.reshape(1, -1)), 
        batch_block_diag(s2_1.reshape(num_sprites, num_feats), a_1.reshape(1, -1)))
  t2 = (batch_block_diag(s1_2.reshape(num_sprites, num_feats), a_2.reshape(1, -1)), 
        batch_block_diag(s2_2.reshape(num_sprites, num_feats), a_2.reshape(1, -1)))

  m1 = custom_get_mask(sprites1, config, a_1)
  m2 = custom_get_mask(sprites2, config, a_2)

  proposals_and_dcs = transitions_and_masks_to_proposals(t1, t2, m1, m2, total_samples, max_ccs)

  res = []
  psprites = copy.deepcopy(sprites2)
  for proposal, dc in proposals_and_dcs:
    psa1, psa2 = proposal
    ps1 = psa1[:-1]
    pa  = psa1[-1:]
    ps2 = psa2[:-1]

    action_idx = len(ps1)
    for idx in dc:
      if idx != action_idx:
        psprites[idx] = copy.deepcopy(sprites1[idx])
    
    # Now we also need to check if the proposal is valid
    pm  = custom_get_mask(psprites, config, pa[0, num_feats:])
    pcc = get_cc_from_mask(pm)
    pdc = disentangled_components(pcc)

    if tuple(dc) in pdc:
      res.append((   ps1[:, :num_feats].flatten(), 
                     pa[0, num_feats:], 
                     reward_fn(ps2[:, :num_feats].flatten()),
                     ps2[:, :num_feats].flatten()  ))

  return res


def relabel_spriteworld(args):
  return relabel_independent_transitions_spriteworld(*args)

def relabel_generic(args):
  return transitions_and_masks_to_proposals(*args)

def enlarge_dataset_spriteworld(data, sprites, config, num_pairs, relabel_samples_per_pair, flattened=True,
                    custom_get_mask=None, pool=True, max_cpus=16):
  data_len = len(data)
  all_idx_pairs = np.array(np.meshgrid(np.arange(data_len), np.arange(data_len))).T.reshape(-1, 2)
  chosen_idx_pairs_idxs = np.random.choice(len(all_idx_pairs), num_pairs)
  chosen_idx_pairs = all_idx_pairs[chosen_idx_pairs_idxs]
  reward_fn = config['task'].reward_of_vector_repr

  config = {
    'action_space': copy.deepcopy(config['action_space']),
    'renderers': {
      'mask': copy.deepcopy(config['renderers']['mask']),
      'mask_abstract': copy.deepcopy(config['renderers']['mask_abstract'])
    }
  }
  args = []
  for (i, j) in chosen_idx_pairs:
    args.append(
        (data[i], sprites[i], data[j], sprites[j], config, reward_fn, relabel_samples_per_pair, flattened, custom_get_mask))

  if pool:
    with mp.Pool(min(mp.cpu_count() - 1, max_cpus)) as pool:
      reses = pool.map(relabel_spriteworld, args)
  else:
    reses = [relabel_spriteworld(_args) for _args in args]
  reses = sum(reses, [])

  return reses

def enlarge_dataset_generic(states, actions, next_states, num_pairs, relabel_samples_per_pair,
                    batch_get_mask=batch_get_heuristic_mask_fetchpush, pool=True, max_cpus=16, add_bad_dcs=False):
  """
  This does random coda on generic data, using the following types:

  Inputs:
    states, next_states: [batch, n_components, state_feats]             # NOTE: assumes 1d state feats
    actions: [batch, action_feats] or [batch, n_components, action_feats]
    custom_get_mask: (batch_states, batched_actions) -> [batch_dim, state_feats + action_feats, state_feats + action_feats]

  Returns:
    (new_states, new_actions, new_next_states) with same shapes as inputs (except batch_dim)
  """
  if pool:
    mp_pool = mp.Pool(min(mp.cpu_count() - 1, max_cpus))
    map_fn  = mp_pool.map
  else:
    map_fn  = map

  data_len, n_state_components, n_state_feats = states.shape
  squeeze_on_return = False
  if len(actions.shape) == 2:
    n_action_components = 1
    squeeze_on_return = True
    actions = actions[:, None]
  else:
    n_action_components = actions.shape[1]

  masks = batch_get_mask(states, actions)
  
  assert masks.shape[1] == masks.shape[2] == n_state_components + n_action_components

  bad_dcs = set([(), tuple(list(range(n_state_components + n_action_components)))])

  Is = np.random.randint(data_len, size=(num_pairs,))
  Js = np.random.randint(data_len, size=(num_pairs,))

  t1s = list(zip(batch_block_diag(states[Is], actions[Is]), batch_block_diag(next_states[Is], actions[Is])))
  t2s = list(zip(batch_block_diag(states[Js], actions[Js]), batch_block_diag(next_states[Js], actions[Js])))

  m1s = masks[Is]
  m2s = masks[Js]

  args = [(t1, t2, m1, m2, relabel_samples_per_pair) for t1, t2, m1, m2 in zip(t1s, t2s, m1s, m2s)]

  proposals_and_dcs = map_fn(relabel_generic, args)
  proposals_and_dcs = sum(proposals_and_dcs, []) # [((sa1, sa2), dc), ... ]

  new_s1s = []
  new_a1s = []
  new_s2s = []
  dcs     = []

  for (sa1, sa2), dc in proposals_and_dcs:
    if (not add_bad_dcs) and (dc in bad_dcs):
      continue
    new_s1s.append(sa1[:n_state_components, :n_state_feats])
    new_a1s.append(sa1[n_state_components:, n_state_feats:])
    new_s2s.append(sa2[:n_state_components, :n_state_feats])
    dcs.append(dc)
  

  new_s1s = np.array(new_s1s)
  new_a1s = np.array(new_a1s)
  new_s2s = np.array(new_s2s)

  if not len(new_s1s):
    if pool:
      mp_pool.close()
      mp_pool.join()
    return (new_s1s, new_a1s, new_s2s)

  masks = batch_get_mask(new_s1s, new_a1s)
  pdcs  = map_fn(get_dcs_from_mask, masks)

  # Now verify that proposals are valid
  valid_idxs = []
  for i, (pdc, dc) in enumerate(zip(pdcs, dcs)):
    if dc in pdc:
      valid_idxs.append(i)

  new_s1s = new_s1s[valid_idxs]
  new_a1s = new_a1s[valid_idxs]
  new_s2s = new_s2s[valid_idxs]

  if squeeze_on_return:
    new_a1s = new_a1s.squeeze(1)

  if pool:
    mp_pool.close()
    mp_pool.join()

  return (new_s1s, new_a1s, new_s2s)
