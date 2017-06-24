from io import BytesIO

from flask import Flask, request, Response, jsonify, send_file
import requests
from PIL import Image

from xvisor import XVisor


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

    score = xvisor.get_preds(img)
    print(score)

    data = {
        'score': score,
        'original': 'http://104.196.225.45:5000/original.jpg',
        'heatmap': 'http://104.196.225.45:5000/heatmap.jpg'
    }
    resp = jsonify(data)
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'

    return resp

@app.route('/original.jpg')
def get_original_img():
    return send_file('original.jpg')

@app.route('/heatmap.jpg')
def get_heatmap_img():
    return send_file('heatmap.jpg')

def img_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('L')
    return img

xvisor = XVisor()

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000, debug=True)
