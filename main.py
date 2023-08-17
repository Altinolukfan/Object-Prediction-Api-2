from flask import Flask, jsonify, request
from keras.models import load_model
from PIL import Image, ImageOps
import urllib.request
import numpy as np
import random
import os

app = Flask(__name__)
class_names=[]
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

f = open("classnames.txt", "r")
readf=f.read()
class_names=readf.split(",")

@app.route('/predict')
def predict():
  link = request.args.get('link')
  #response = request.get(link)
  num = random.random()
  urllib.request.urlretrieve(link, str(num)+".png")
  img=str(num)+".png"
  image = Image.open(str(img)).convert('RGB')
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)
  image_array = np.asarray(image)
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  data[0] = normalized_image_array
  prediction = model.predict(data)
  #print(prediction)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]
  os.remove(str(num)+".png")
  return jsonify(confidence_score)

if __name__=='__main__':
    app.run(debug=True)
