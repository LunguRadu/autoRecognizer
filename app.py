import base64
import json
import numpy as np
import io
import re
from PIL import Image
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request, render_template
from flask import jsonify
from flask import Flask
from flask_cors import CORS



app = Flask(__name__)
CORS(app)


# Function to get the model
def get_model():
    global model
    model = load_model('fine_tuning_vgg16_1.h5')
    # model = load_model('autovgg15model.h5')
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

    # message = request.get_json(force=True)
    # encoded = message['image']
    # image_data = re.sub('^data:image/.+;base64,', '', encoded)
    # decoded = base64.b64decode(image_data)
    # image = Image.open(io.BytesIO(decoded))
    # processed_image = preprocess_image(image, target_size=(224, 224))
    # class_index = np.argmax(model.predict(processed_image), axis=-1)[0]
    # # prediction = get_class(class_index)
    # response = {"prediction": get_class(class_index)}
    # # return render_template("predict.html", prediction=prediction)
    # return jsonify(response)

    NUMBER_PREDICTION = 5
    message = request.get_json(force=True)
    encoded = message['image']
    image_data = re.sub('^data:image/.+;base64,', '', encoded)
    decoded = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(processed_image)[0]
    top_5_indices = (-prediction).argsort()[:NUMBER_PREDICTION]
    prediction_confidence = get_prediction_confidence(top_5_indices, prediction, NUMBER_PREDICTION)
    response = {"prediction":str(prediction_confidence)}
    return json.dumps(response)

def get_prediction_confidence(class_indices, prediction_percentage, NUMBER_PREDICTION):
    with open('bodyType_classes.json') as json_file:
        classes_dic = json.load(json_file)
        result = dict()
        for index in range(NUMBER_PREDICTION): # 0 --> 5 top 5 prediction
            class_index = class_indices[index]
            result[classes_dic[str(class_index)]] = prediction_percentage[class_index]
        return result

def get_class(class_index):
    with open('bodyType_classes.json') as json_file:
        classes_dic = json.load(json_file)
        return classes_dic[str(class_index)]


if __name__ == '__main__':
    Flask.run(app)
