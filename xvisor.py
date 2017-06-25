import os
import random

import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np

from xnet import XNet
from vision_utils import preprocess
import config


class XVisor(object):
    def __init__(self):
        self.xnet = XNet().cuda()
        self.xnet.eval()
        self.xnet.load_state_dict(torch.load(config.MODEL_PATH))

    def get_preds(self, img):
        '''
        img: 1 x 1024 x 1024 FloatTensor
        '''
        req_id = random.randint(100000, 999999)
        original_path = os.path.join(config.IMG_PATH, 'original{}.jpg'.format(req_id))
        heatmap_path = os.path.join(config.IMG_PATH, 'heatmap{}.jpg'.format(req_id))

        X = preprocess(img)
        X = X.unsqueeze(0)
        X_var = Variable(X.cuda(), requires_grad=True)

        score_var = self.xnet(X_var)
        score = float(score_var.data.cpu().numpy().flatten()[0])

        img_crop = X.squeeze().unsqueeze(-1).repeat(1, 1, 3).numpy()
        Image.fromarray(np.uint8(img_crop * 256)).save(original_path)
        
        img_crop *= 128

        score_var.backward()
        img_grad = X_var.grad.data.squeeze().cpu().numpy()

        x, y = img_grad.shape
        overlay = np.zeros([x, y, 3])
        overlay[:, :, 0] = abs(img_grad) ** 0.5
        overlay[:, :, 0] *= 256 / overlay.max()

        heatmap = overlay + img_crop
        heatmap = np.uint8(heatmap)

        heatmap_img = Image.fromarray(heatmap)
        heatmap_img.save(heatmap_path)

        return score, original_path, heatmap_path
