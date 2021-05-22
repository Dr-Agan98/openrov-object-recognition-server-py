from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from flask import request
from PIL import Image
import base64, io
import numpy as np
from yolov4 import Detector

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

d = Detector(config_path="/home/nagard/Progetti/openrov-object-recognition-server-py/yolov4-tiny-3l.cfg", weights_path="/home/nagard/Progetti/openrov-object-recognition-server-py/yolov4-tiny-3l_best_2.weights" , meta_path="/home/nagard/Progetti/openrov-object-recognition-server-py/data/detector.data")

#@app.route("/",methods=['GET','POST'])
#def root():
#    return '{box:null}'

@socketio.on("test")
def handle_my_custom_event(data):
    img = Image.open(io.BytesIO( base64.b64decode( data['data'].split(',')[1] )))
    img_arr = np.array(img.resize((d.network_width(), d.network_height())))
    detections = d.perform_detect(image_path_or_buf=img_arr, show_image=False)
    for detection in detections:
        box = detection.left_x, detection.top_y, detection.width, detection.height
        print(f'{detection.class_name.ljust(10)} | {detection.class_confidence * 100:.1f} % | {box}')
    

socketio.run(app, host="0.0.0.0")