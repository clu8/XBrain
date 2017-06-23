import os
from collections import Counter

import dicom

import config


path = os.path.join(config.DATA_PATH, config.DATA_FOLDER)
dicom_files = [os.path.join(path, filename) for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename))]
c = Counter()

for df in dicom_files:
    ds = dicom.read_file(df)
    x, y = ds.pixel_array.shape
    print(x, y, x / y)
    c[(x, y)] += 1

print(c.most_common())