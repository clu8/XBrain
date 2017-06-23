import os
import pickle

from PIL import Image
import numpy as np

import config


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
            X = np.array(img.getdata())
            y = get_label(f)
            if i % 10 == 0:
                X_test.append(X)
                y_test.append(y)
            else:
                X_train.append(X)
                y_train.append(y)

print('Num train: {}'.format(len(X_train)))
print('Num test: {}'.format(len(X_test)))

with open(config.PROCESSED_PATH, 'wb') as f:
    pickle.dump((X_train, y_train, X_test, y_test), f, protocol=pickle.HIGHEST_PROTOCOL)
