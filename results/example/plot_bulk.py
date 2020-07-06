import glob
import os
import ast
import numpy as np
from copy import deepcopy
import seaborn as sns; sns.set()
sns.set_style('whitegrid')
from scipy.signal import medfilt

import argparse

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12})


class AttrDict(dict):
  __setattr__ = dict.__setitem__

  def __getattr__(self, key):
    try:
      return dict.__getitem__(self, key)
    except KeyError:
      raise AttributeError

def smooth_reward_curve(x, y):
   halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
   k = halfwidth
   xsmoo = x
   ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
       mode='same')
   return xsmoo, ysmoo

def fix_point(x, y, interval):
  """What does this do?"""
  np.insert(x, 0, 0)
  np.insert(y, 0, 0)

  fx, fy = [], []
  pointer = 0
  ninterval = int(max(x) / interval + 1)

  for i in range(ninterval):
    tmpx = interval * i

    while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
      pointer += 1

    if pointer + 1 < len(x):
      alpha = (y[pointer + 1] - y[pointer]) / (x[pointer + 1] - x[pointer])
      tmpy = y[pointer] + alpha * (tmpx - x[pointer])
      fx.append(tmpx)
      fy.append(tmpy)

  return fx, fy

def load_data(indir, smooth, bin_size):
  datas = []
  infiles = glob.glob(os.path.join(indir, '*.csv'))
  # datas_goal = []

  for inf in infiles:
    data = []
    data_csv = np.loadtxt(inf, delimiter=",", skiprows=1, usecols=[1, 2])
    for sec, acc in zip(data_csv[:, 0], data_csv[:, 1]):
      data.append([sec, acc])
    datas.append(data)

  def process_data(datas):
    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for data in datas:
      result.append([timesteps, data[-1]])
      timesteps = data[0]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
      x, y = smooth_reward_curve(x, y)

    if smooth == 2:
      y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]
    
  # if goal:
  #   return list(zip(*tuple([process_data(data_goal[goal]) for data_goal in datas_goal])))

  # else:

  return list(zip(*(process_data(data) for data in datas)))

def load(indir, names, labels, smooth, bin_size):
  result = []

  for name, label in zip(names, labels):
    d = os.path.join(indir, name)
    tmpx, tmpy = [], []
    
    tx, ty = load_data(d, smooth, bin_size)
    tmpx += tx
    tmpy += ty

    if len(tmpx) > 1:
      length = min([len(t) for t in tmpx])
      for i in range(len(tmpx)):
        tmpx[i] = tmpx[i][:length]
        tmpy[i] = tmpy[i][:length]

      x = np.mean(np.array(tmpx), axis=0)
      y_mean = np.mean(np.array(tmpy), axis=0)
      y_std = np.std(np.array(tmpy), axis=0)
    else:
      x = np.array(tmpx).reshape(-1)
      y_mean = np.array(tmpy).reshape(-1)
      y_std = np.zeros(len(y_mean))

    result.append([label, x, y_mean, y_std])
  return result


def plot(args):
  plt.figure(figsize=(4,3), dpi=300)
  sns.despine(left=True, bottom=True)
  datas = load(args.source, args.names, args.labels, args.smooth, args.bin_size)
  for ignore_string in args.ignore_strings:
    datas = [d for d in datas if not ignore_string in d[0]]
  lines = []
  max_y = args.max_y
  min_y = args.min_y
  max_x = args.max_x
  min_x = 1e10

  for i, data in enumerate(datas):
    label, x, y_mean, y_std = data
    color = args.colors[i]
    if np.sum(y_std):
      y_upper = y_mean + y_std
      y_lower = y_mean - y_std
      plt.fill_between(
          x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.2
      )
    linestyle = args.line_styles[i]
    marker = args.markers[i]
    markersize = args.markersizes[i]
    markevery=int(max_x/500)
    line = plt.plot(x, list(y_mean), linewidth=2.0, label=label, color=color, markersize=markersize, marker=marker, markevery=markevery, linestyle=linestyle)
    lines.append(line[0])
    if max(x) < min_x:
      min_x = max(x)


  xticks = np.arange(0, max_x+1, max_x // 8 )
  ignore_even = lambda s, i: s if i %2 == 0 else ''
  if 'Millions' in args.xlabel:
    xlabels = [ignore_even('{:0.2f}'.format(float(x) / 100000000. * args.time_steps_per_episode).rstrip('0').rstrip('.'),
                i) for i, x in enumerate(xticks)]
  elif 'Thousands' in args.xlabel:
    xlabels = [ignore_even('{:0.2f}'.format(float(x) / 100000. * args.time_steps_per_episode).rstrip('0').rstrip('.'),
                i) for i, x in enumerate(xticks)]
  #xlabels[-1] += 'M'
  plt.xticks(xticks, xlabels, fontsize=args.label_size)
  if args.xlabel is None:
    plt.xlabel('Placeholder', fontsize=args.label_size, color='white')
  else:
    plt.xlabel(args.xlabel, fontsize=args.label_size)

  if hasattr(args, 'y_values'):
    plt.yticks(args.y_values, args.y_labels, fontsize=args.label_size)
  if args.ylabel is None:
    plt.ylabel('')
  else:
    plt.ylabel(args.ylabel, fontsize=args.label_size)
  
  plt.ylim(min_y, max_y)
  plt.xlim(0, max_x)
  plt.legend(fancybox=True, framealpha=0.7, loc=args.legend_loc, fontsize=args.legend_size, frameon=True, facecolor='white', borderaxespad=1.)
  plt.title(args.title, fontsize=args.title_size)
  plt.tight_layout(pad=0.0) # Make room for the xlabel
  plt.savefig(args.output, format='png', dpi=600) # Need to do savefig before plt.show() otherwise PDF file will be blank
  print("DONE {}".format(args.output))

if __name__ == '__main__':

  with open(os.path.join('confs','shared.conf')) as f:
    shared = dict((l.split('%')[0].split('=') for l in f.readlines() if l and not l[0] == '%'))

  for conf in glob.glob('confs/*.conf'):
    
    # ignore shared conf
    if 'shared' in conf:
      continue

    with open(conf) as f:
      b =  dict((l.split('%')[0].split('=') for l in f.readlines() if l and not l[0] == '%'))
    
    a = deepcopy(shared)
    a.update(b)

    for k, v in a.items():
      a[k] = ast.literal_eval(v)
    
    a = AttrDict(a)

    a.output = os.path.join('plots', os.path.splitext(os.path.basename(conf))[0]) + '.png'
  
    print(conf)
    plot(a)