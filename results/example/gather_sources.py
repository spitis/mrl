"""Based on what already exists in sources directory,
this updates it from a source folder (exported from Tensorboard)"""

import sys, glob, os
from shutil import copyfile

csv_folder = sys.argv[1]
sources = glob.glob('sources/**/*.csv', recursive=True)

for source in sources:
  filename = os.path.basename(source)
  csv = os.path.join(csv_folder, filename)
  if os.path.exists(csv):
    copyfile(csv, source)