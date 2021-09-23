from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO,emit
from PIL import Image
import base64, io
import numpy as np
import json
import darknet, darknet_images
import cv2

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

net, class_names, colors = darknet.load_network('./yolov4-tiny-3l.cfg', './data/detector.data', './yolov4-tiny-3l_best_2.weights')
net_width = darknet.network_width(net)
net_height = darknet.network_height(net)

@app.route("/",methods=['GET','POST'])
def root():
    return '{box:null}'

@socketio.on("test")
def handle_my_custom_event(data):
    imgURI = json.loads(data)
    img = Image.open(io.BytesIO( base64.b64decode( imgURI['data'].split(',')[1] )))
    img_width = img.width
    img_height = img.height
    img_arr = np.array(img)

    img, detections  = darknet_images.detect(img_arr, net, class_names, colors, thresh=0.1)

    bd_boxs = []
    for label, conf, coords in detections:
        x, y, w, h = coords
        bd_box = {
                'class': label,
                'confidence': float(conf),
                'box':{
                    'x': int((x*img_width) / net_width),
                    'y': int((y*img_height) / net_height),
                    'w': int((w*img_width) / net_width),
                    'h': int((h*img_height) / net_height)
                }       
        }
        bd_boxs.append(bd_box)
        #print(bd_box)

    emit("detection", json.dumps(bd_boxs))

#cv2.imwrite("./prediction.jpeg", img)

socketio.run(app, host="0.0.0.0")