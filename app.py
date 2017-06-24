from io import BytesIO

from flask import Flask, request, jsonify
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
    print('here')
    return 'Welcome to XVisor'

@app.route('/test')
def test():
    print('test')
    return 'testing'

@app.route('/scan', methods=['GET'])
def scan():
    print(request.args)
    img_url = request.args.get('url')
    print('Scan called with URL: {}'.format(img_url))
    score = float(get_preds(img_from_url(img_url))[0])
    print(score)

    data = {
        'score': score
    }
    resp = jsonify(data)
    resp.status_code = 200

    return resp

def img_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('L')
    return img

def get_preds(img):
    X = preprocess(img)
    X = X.unsqueeze(0)
    X_var = Variable(X.cuda(), volatile=True)
    score_var = xnet(X_var)
    score = score_var.data.cpu().numpy().flatten()
    return score

xnet = XNet().cuda()
xnet.eval()
xnet.load_state_dict(torch.load(config.MODEL_PATH))

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000, debug=True)
