from flask import Flask, jsonify, request
from keras.models import load_model
from PIL import Image, ImageOps
import urllib.request as rq
import numpy as np
import random
import os
import requests

#from OpenSSL import SSL
#context = SSL.Context(SSL.PROTOCOL_TLSv1_2)

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
  num = random.random()
  response = requests.get(str(link))
  with open(str(num)+".png", "wb") as f:
      f.write(response.content)
      f.close()
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
  return jsonify(classname=class_name,confidence=confidence_score)

@app.route('/')
def hello_world():
  return 'Hello, World!'

#if __name__=='__main__':
    #app.run(debug=True,ssl_context=context)
