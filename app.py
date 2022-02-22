import base64
import json
from unicodedata import name
import logging

import numpy as np
import io
import re
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request, render_template
from flask import Flask
from flask_cors import CORS
from google.cloud import storage

app = Flask(__name__)
"""
For deployment: 
change the CORS policy below to heroku frontend only 
"""
## For produciton 
# CORS(app, origins=["https://cs484-cnn-classifier.herokuapp.com/:7800"])
## For development 
CORS(app)


# Function to get the model
def get_model():
    global model
    storage_client = storage.Client.from_service_account_json(
        'google-credentials.json')
    bucket = storage_client.bucket("vgg16_auto_model")
    model_vgg = bucket.get_blob("autovgg15model.h5")
    model_vgg.download_to_filename("fine_tuning_body_v1.h5")
    model = load_model('fine_tuning_body_v1.h5')
    print(" * Model loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


print(" * Loading Keras model...")
get_model()


@app.route("/predict", methods=["POST"])
def predict():

    # Flask version
    # message = request.get_json(force=True)
    # encoded = message['image']
    # image_data = re.sub('^data:image/.+;base64,', '', encoded)
    # decoded = base64.b64decode(image_data)
    # image = Image.open(io.BytesIO(decoded))
    # processed_image = preprocess_image(image, target_size=(224, 224))
    # class_index = np.argmax(model.predict(processed_image), axis=-1)[0]
    # # prediction = get_class(class_index)
    # response = {"prediction": get_class(class_index)}
    # return jsonify(response)
    # get_model()

    # Svelte version
    NUMBER_PREDICTION = 5
    message = request.get_json(force=True)
    encoded = message['image']
    image_data = re.sub('^data:image/.+;base64,', '', encoded)
    decoded = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(decoded))
    #  Preprocess image for Models
    processed_image = preprocess_image(image, target_size=(224, 224))
    # Model order array
    response = getPrediction(model,processed_image, NUMBER_PREDICTION)
    return json.jsonify(response)


def getPrediction(model,  preprocessed_image, number_top_prediction):
    #  get prediction from model
    prediction = model.predict(preprocessed_image)[0]
    #  get top indice associate to classification
    top_indices = (-prediction).argsort()[:number_top_prediction]
    #  map class and prediction confidence 
    prediction_confidence, classification_order = get_prediction_confidence(top_indices, prediction, number_top_prediction)
    return {"confidence":prediction_confidence, "order": classification_order}

def get_prediction_confidence(class_indices, prediction_percentage, NUMBER_PREDICTION):
    with open('res/bodyType_classes.json') as json_file:
        classes_dic = json.load(json_file)
        result = dict()
        order = []
        for index in range(NUMBER_PREDICTION): # 0 --> 5 top 5 prediction
            class_index = class_indices[index]
            class_name = classes_dic[str(class_index)]
            result[class_name] = str(prediction_percentage[class_index])
            order.append(class_name)
        return result, order

# def get_class(class_index):
#     with open('bodyType_classes.json') as json_file:
#         classes_dic = json.load(json_file)
#         return classes_dic[str(class_index)]


if __name__ == '__main__':
    app.run()
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
