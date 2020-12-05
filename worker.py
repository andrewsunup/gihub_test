from __future__ import division, print_function
import os
import numpy as np
import requests
from PIL import Image
import torch.nn.functional as F
from torchvision import models, transforms
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
from flask import request, Flask, render_template, jsonify
import time

status = "Idel"
result = ""

# data preprocess
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225])
])

# get the labels
LABELS_URL = 'https://s3.amazonaws.com/mlpipes/pytorch-quick-start/labels.json'
labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}

# get the model
model = models.vgg16(pretrained=True).eval()

app = Flask(__name__)
@app.route('/')
def home():
    return "This is a worker."

@app.route('/status', methods = ['GET'])
def getStatus():
    return status

@app.route('/predict', methods = ['POST'])
def predict():

    # record the start time
    start_time = time.time()

    # get the image file
    f = request.files['image']

    # save the file
    root = os.path.dirname(__file__)
    img_folder = os.path.join(root, 'images')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img_path = os.path.join(img_folder, secure_filename(f.filename))
    f.save(img_path)

    # recognize the image
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    fc_out = model(img_tensor)
    fc_out = F.softmax(fc_out)
    lbl_out = np.argsort(fc_out[0].data.numpy())[-1]

    result = " Top-1 prediction: " + str(labels[lbl_out]) #" \";  Confidence: %.3f" % (fc_out[0].data.numpy()[lbl_out])

    print("prediction finished")
    print(result)

    return jsonify({'name': "test_name", 'result': result})


@app.route('/result', methods = ['GET'])
def getResult():
    return result

if __name__ == "__main__":

    # poke the manager
    url = 'http://192.168.68.61:5555/addnode'
    r = requests.get(url)

    app.run(host = "0.0.0.0", port = "5000")
    #http_server = WSGIServer(('127.0.0.1', 5000), app)
    #http_server.serve_forever()
