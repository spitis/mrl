import glob
import pandas
import numpy as np
from collections import defaultdict
from sklearn.utils import resample

def parse_title(csv_filename, items=['seed', 'real', 'coda', 'noise', 'dyna', 'roll', 'mbpo', 'c3xm']):
  f = csv_filename.split('/')[-1]
  res = []
  for item in items:
    if item in f:
      res.append(f.split(item)[-1].split('_')[0])
    else:
      res.append(None)
  return res

def make_latex_table(args):
  csvs = glob.glob(f'{args.results_folder}/**/*Success.csv')

  results = defaultdict(lambda: defaultdict(list))
  for csv in csvs:
    seed, real, coda, _, dyna, roll, mbpo, c3xm = parse_title(csv)
    coda_to_real_ratio = int(coda) // int(real)
    csv = pandas.read_csv(csv)
    dyna = dyna and int(dyna)
    mbpo = mbpo and int(mbpo)
    c3xm = c3xm and int(c3xm)
    if c3xm:
      mbpo = c3xm
    if dyna:
      coda_to_real_ratio = f'{coda_to_real_ratio}_{roll}'
    if mbpo:
      mbpo_to_real_ratio = int(mbpo) // int(real)
      if c3xm:
        coda_to_real_ratio = f'{coda_to_real_ratio}_{mbpo_to_real_ratio}_{roll}_c3'
      else:
        coda_to_real_ratio = f'{coda_to_real_ratio}_{mbpo_to_real_ratio}_{roll}'
    BASE = 500
    results[real][coda_to_real_ratio].append(csv['Test/Success'][BASE-20:BASE].mean())

  for datasize in ['25000','50000','75000','100000','150000','250000']:
    hphantom=''
    if len(datasize) == 5:
      hphantom='\hphantom{0}'
    s = [f'${hphantom}{int(datasize)//1000}$','&']
    # 0_1 : Dyna 1 step
    # 0_1_1 : MBPO 1 step
    # 0_1_5 : MBPO 5 step
    # 3_1_1 : CODA+MBPO 1 step
    # 3_1_5 : CODA+MBPO 5 step
    #for coda_to_real_ratio in [0, 1, 3, 5]:
    means = []
    stds = []
    #for coda_to_real_ratio in [0, 1, 3, 5, '0_1_5', '0_1_5_c3']:
    for coda_to_real_ratio in [0, '0_1', '0_1_5', 1, 2, 3, 5, '3_1_5']:
      values = results[datasize][coda_to_real_ratio]
      if len(values) not in [5,10]: # TODO If more/less than 10 seeds are run, change this. 
        mean = -1
        std = -1
      else:
        values = np.array(values)*100
        mean = (np.mean(values)).round(1)
        # BOOTSTRAP THE STANDARD ERROR
        resampled_means = []
        for i in range(1000):
          resampled_means.append(np.mean(resample(values)))
        std = np.std(resampled_means).round(1)
      hphantom=''
      #if len(str(std).split('.')[0]) == 1:
      #  hphantom='\hphantom{0}'
      means.append(mean)
      stds.append(std)
    max_mean = max(means)
    for (mean, std) in zip(means, stds):
      m = f'\mybm{{{mean}}}' if mean == max_mean else mean
      s += [f'${m} \pm {hphantom}{std}$ ', '&']
    s = s[:-1] + ['\\\\']
    print(*s)

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(
      description="Make Latex table for BatchRL experiment",
      formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--results_folder', type=str)

  args = parser.parse_args()
  make_latex_table(args)