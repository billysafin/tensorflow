#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import helper

sys.path.append(os.getcwd() + '/..')

import learning.image.predict as predictor

from flask import Flask
from flask import request
from flask import render_template
from werkzeug import secure_filename

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['jpeg','jpg', 'png'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    title = "Welcome to Le Creuset Image Recongiation Page"
    message = "Welcome to Le Creuset Image Recongiation Page"
    
    return render_template('index.html', message=message, title=title)

@app.route('/post', methods=['POST'])
def image_recongization():
    title = "image recongization"
    
    f = request.files['img_file']
    how_to_predict = int(request.form['how_to_predict'])
    
    #uploading image
    prediction = None
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        path_2_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    
        if not os.path.exists(path_2_tmp):
            os.mkdir(path_2_tmp)
        f.save(os.path.join(path_2_tmp, filename))
        
    #image recongization    
    image_path = os.path.join(path_2_tmp, filename)
    if how_to_predict == 1:
        pb_path = helper.get_latest_modified_file_path(helper.learned_log_dir, ext='.pb')
        
        prediction = predictor.main.predictUsingPb(image_path, pb_path)
    elif how_to_predict == 2:
        prediction = predictor.main.predictUsingCkpt(image_path, helper.learned_log_dir)
        
    if prediction in helper.LABELS:
        prediction = helper.LABELS[prediction]
    else:
        prediction = 'Failuer'

    return render_template('index.html', title=title, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)