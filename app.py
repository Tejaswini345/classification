import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
import scipy
import tensorflow as tf


app = Flask(__name__)


model = load_model('./DensenetModel.h5')
classes = ['MildDemented', 'ModerateDemented',
           'NonDemented', 'VeryMildDemented']


def getResult(img):
    img = tf.keras.utils.load_img(img, target_size=(229, 229))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    result = model.predict(img_array)
    dict_result = {}
    score = tf.nn.softmax(result[0])
    for i in range(4):
        dict_result[classes[i]] = score[i]

    max_res = max(dict_result, key=dict_result.get)

    return max_res


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/check_input', methods=['POST'])
def check_input():
    input_value = request.form['pw']

    if input_value == "12345":
        return render_template('next_page.html')
    else:
        return render_template('incorrect_page')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'dataset/train', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        return value
    return None


if __name__ == '__main__':
    app.run(debug=True)
