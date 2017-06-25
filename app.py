from io import BytesIO

from flask import Flask, request, Response, jsonify, send_from_directory
import requests
from PIL import Image

from xvisor import XVisor
import config


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
    img_url = request.args.get('url')
    print('Scan called with URL: {}'.format(img_url))

    if img_url is None:
        err = 'No URL provided'
        print(err)
        return Response(response=err, status=400)

    try:
        img = img_from_url(img_url)
    except:
        err = 'Image load error'
        print(err)
        return Response(response=err, status=400)

    score, original_path, heatmap_path = xvisor.get_preds(img)
    print(score, original_path, heatmap_path)

    data = {
        'score': score,
        'original': 'http://104.196.225.45:5000/{}'.format(original_path),
        'heatmap': 'http://104.196.225.45:5000/{}'.format(heatmap_path)
    }
    resp = jsonify(data)
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'

    return resp

@app.route('/img/<path:filename>')
def get_img(filename):
    print('Getting image: {}'.format(filename))
    return send_from_directory(config.IMG_PATH, filename)

def img_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('L')
    return img

xvisor = XVisor()

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000, debug=True, threaded=True)
