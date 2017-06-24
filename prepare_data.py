import os
import pickle

import torch
from PIL import Image
import numpy as np

import config
from vision_utils import preprocess


def is_img_file(filename):
    return os.path.isfile(os.path.join(path, filename)) and os.path.splitext(filename)[1] == '.png'

def get_label(filename):
    assert filename[-4:] == '.png'

    if filename[-5] == '1':
        return 1
    elif filename[-5] == '0':
        return 0
    else:
        raise ValueError('Invalid filename label format')

X_train = []
y_train = []
X_test = []
y_test = []

for path in (config.SHENZHEN_PATH, config.MONTGOMERY_PATH):
    img_files = [os.path.join(path, filename) for filename in os.listdir(path) if is_img_file(filename)]

    for i, f in enumerate(img_files):
        print(f)
        with Image.open(f) as img:
            img = img.convert('L')

            X = preprocess(img)
            y = get_label(f)

            if i % 10 == 0:
                X_test.append(X)
                y_test.append(y)
            else:
                X_train.append(X)
                y_train.append(y)

X_train = torch.stack(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.stack(X_test)
y_test = torch.FloatTensor(y_test)

print(X_train.size())
print(y_train.size())
print(X_test.size())
print(y_test.size())

with open(config.PROCESSED_PATH, 'wb') as f:
    pickle.dump((X_train, y_train, X_test, y_test), f, protocol=pickle.HIGHEST_PROTOCOL)

with open(config.SMALL_PATH, 'wb') as f:
    pickle.dump((X_train[:10], y_train[:10], X_test[:2], y_test[:2]), f, protocol=pickle.HIGHEST_PROTOCOL)
