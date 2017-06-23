import os


DATA_PATH = '../datasets'

SHENZHEN_PATH = os.path.join(DATA_PATH, 'ChinaSet_AllFiles', 'CXR_png')
MONTGOMERY_PATH = os.path.join(DATA_PATH, 'MontgomerySet', 'CXR_png')

PROCESSED_PATH = os.path.join(DATA_PATH, 'processed.pickle')
SMALL_PATH = os.path.join(DATA_PATH, 'processed_small.pickle')
