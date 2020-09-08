# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:33:38 2020

@author: sys
"""

import numpy as np
import os
#from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
global sess
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
app = Flask(__name__)
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
model = load_model(r"c:\Users\sys\conversation engine.h5", compile=False)
@app.route('/')
def index():
    return render_template('page.html')
@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        img = image.load_img(filepath,target_size = (64,64)) 
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        with graph.as_default():
            set_session(sess)
            preds = model.predict_classes(x)
    
            print("prediction",preds)
        index = ['zero','one','two','three','four','five','six','seven','eight','nine','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        text = "the predicted hand gesture is : " + str(index[preds[0]])
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)