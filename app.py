from io import BytesIO

from flask import Flask
import requests
from PIL import Image
import torch
from torch.autograd import Variable

from xnet import XNet
import config
from vision_utils import preprocess


app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to XBrain'

@app.route('/scan', methods=['POST'])
def scan():
    pass

def img_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('L')
    return img

def get_preds(img):
    X = preprocess(img)
    X = X.unsqueeze(0)
    X_var = Variable(X.cuda(), volatile=True)
    score = xnet(X_var)
    print(score)

xnet = XNet().cuda()
xnet.eval()
xnet.load_state_dict(torch.load(config.MODEL_PATH))
