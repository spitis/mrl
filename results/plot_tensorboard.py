import glob
import os
import ast
import numpy as np
import seaborn as sns; sns.set()
sns.set_style("whitegrid")
from scipy.signal import medfilt

import argparse

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12})

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

def load(indir, smooth, bin_size):
  result = []

  for d in sorted(glob.glob(os.path.join(indir, '*'))):
    if not os.path.isdir(d):
      continue
    tmpx, tmpy = [], []
    label = d.strip('/').split('/')[-1]

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

# COLORS = [
#   '#1f77b4',  # muted blue
#   '#ff7f0e',  # safety orange
#   '#2ca02c',  # cooked asparagus green
#   '#d62728',  # brick red
#   '#9467bd',  # muted purple
#   '#8c564b',  # chestnut brown
#   '#e377c2',  # raspberry yogurt pink
#   '#7f7f7f',  # middle gray
#   '#bcbd22',  # curry yellow-green
#   '#17becf'  # blue-teal
# ]

# COLORS = [
#   '#1f77b4',  # muted blue
#   '#1f77b4',  # safety orange
#   '#d62728',  # cooked asparagus green
#   '#d62728',  # brick red
#   '#666666',  # muted purple
#   '#666666',  # chestnut brown
#   '#e377c2',  # raspberry yogurt pink
#   '#7f7f7f',  # middle gray
#   '#bcbd22',  # curry yellow-green
#   '#17becf'  # blue-teal
# ]

COLORS = [
  '#1f77b4',  # muted blue
  #'#1f77b4',  # safety orange
  '#d62728',  # cooked asparagus green
  #'#d62728',  # brick red
  '#666666',  # muted purple
  '#666666',  # chestnut brown
  '#e377c2',  # raspberry yogurt pink
  '#7f7f7f',  # middle gray
  '#bcbd22',  # curry yellow-green
  '#17becf'  # blue-teal
]

def plot(args):
  plt.figure(figsize=(4,3.5), dpi=200)
  sns.despine(left=True, bottom=True)
  datas = load(args.source, args.smooth, args.bin_size)
  lines = []
  max_y = args.max_y
  min_y = args.min_y
  max_x = args.max_x
  min_x = 1e10

  for i, data in enumerate(datas):
    label, x, y_mean, y_std = data
    color = COLORS[i]
    if np.sum(y_std):
      y_upper = y_mean + y_std
      y_lower = y_mean - y_std
      plt.fill_between(
          x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
      )
    if args.line_styles is not None:
      linestyle = args.line_styles[i]
      #if i % 2 == 1:
      #  line = plt.plot(x, list(y_mean), linewidth=1.0, label=label, color=color, markersize=8.0, marker="*", markevery=int(max_x/500), linestyle=linestyle)
      #else:
      line = plt.plot(x, list(y_mean), linewidth=1.0, label=label, color=color, linestyle=linestyle)
    else:
      line = plt.plot(x, list(y_mean), label=label, color=color)
    lines.append(line[0])
    if max(x) < min_x:
      min_x = max(x)

  if hasattr(args, 'y_values'):
    plt.yticks(args.y_values, args.y_labels)

  plt.xticks(args.x_values, args.x_labels)
  if args.xlabel is None:
    plt.xlabel('Epoch')
  else:
    plt.xlabel(args.xlabel)

  if args.ylabel is None:
    plt.ylabel('Success Rate')
  else:
    plt.ylabel(args.ylabel)
  
  plt.ylim(min_y, max_y)
  plt.xlim(0, max_x)
  plt.legend(loc=args.legend_loc, prop={'size': args.legend_size})
  plt.title(args.title)
  plt.tight_layout(pad=0.0) # Make room for the xlabel
  plt.savefig(args.output, format='pdf', dpi=100) # Need to do savefig before plt.show() otherwise PDF file will be blank
  plt.show()
  plt.draw()
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run commands")
  parser.add_argument('-s', '--source', type=str, default="", help="Folder with results", required=True)
  parser.add_argument('--smooth', type=int, default=1 , help="1 = ???, 2 = Smooth with openai smoothing")
  parser.add_argument('-b', '--bin_size', type=int, default=25, help="bin size for average")
  parser.add_argument('-o', '--output', type=str, default="", help='outfile')
  parser.add_argument('-l', '--legend_size', type=int, default=12, help='font size for legend')

  args = parser.parse_args()
  
  if not args.output:
    args.output = 'output_{}.pdf'.format(args.source.strip('/').split('/')[-1])
  
  with open(os.path.join(args.source, 'plot.conf')) as f:
    argdict = dict((l.split('=') for l in f.readlines() if l))
  
  for k, v in argdict.items():
    setattr(args, k, ast.literal_eval(v))
  
  plot(args)
