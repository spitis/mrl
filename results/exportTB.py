import tensorflow as tf
import time
import csv
import sys
import os
import collections
import glob
import tqdm

# Import the event accumulator from Tensorboard. Location varies between Tensorflow versions. Try each known location until one works.
eventAccumulatorImported = False;
# TF version < 1.1.0
if (not eventAccumulatorImported):
  try:
    from tensorflow.python.summary import event_accumulator
    eventAccumulatorImported = True;
  except ImportError:
    eventAccumulatorImported = False;
# TF version = 1.1.0
if (not eventAccumulatorImported):
  try:
    from tensorflow.tensorboard.backend.event_processing import event_accumulator
    eventAccumulatorImported = True;
  except ImportError:
    eventAccumulatorImported = False;
# TF version >= 1.3.0
if (not eventAccumulatorImported):
  try:
    from tensorboard.backend.event_processing import event_accumulator
    eventAccumulatorImported = True;
  except ImportError:
    eventAccumulatorImported = False;
# TF version = Unknown
if (not eventAccumulatorImported):
  raise ImportError('Could not locate and import Tensorflow event accumulator.')

def exitWithUsage():
	print(' ')
	print('Usage:')
	print('   python exportTB.py <output-folder> <output-path-to-csv> <summaries>')
	print('Inputs:')
	print('   <logfolder>              - Path to log folder.')
	print('   <output-folder>          - Path to output folder.')
	print('   <tags>                   - Comma separated list of tags to save')
	print(' ')
	sys.exit()

if (len(sys.argv) < 3):
	exitWithUsage()

logdir = sys.argv[1]
if not logdir[-1] == '/':
  logdir += '/'
outputFolder = sys.argv[2]

if not os.path.isdir(outputFolder):
  os.makedirs(outputFolder)

run_names = glob.glob(logdir + '**/events.out*', recursive=True)
tag_names = summaries = sys.argv[3].split(',')

for run in tqdm.tqdm(run_names):
  ea = event_accumulator.EventAccumulator(run,
    size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 0, # 0 = grab all
        event_accumulator.HISTOGRAMS: 1,
  })
  ea.Reload()
  
  for tag in tag_names:
    try:
      res = ea.Scalars(tag)
      csvFileName = os.path.join(outputFolder, run.split('/')[-2] + '_' + tag.replace('/','___') + '.csv')

      with open(csvFileName, 'w') as csvfile:
        logWriter = csv.writer(csvfile, delimiter=',')

        # Write headers to columns
        headers = ['wall_time','step', tag]
        logWriter.writerow(headers)
    
        vals = ea.Scalars(tag)
        for i in range(len(vals)):
          v = vals[i]
          data = [v.wall_time, v.step, v.value]
          logWriter.writerow(data)
    except:
      print("Failed to find tag {} in {}".format(tag, run))
      continue