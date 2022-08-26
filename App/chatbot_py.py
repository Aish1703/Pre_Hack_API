import random
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, request
import nltk
import numpy as np
from keras.models import model_from_json
import requests
from io import BytesIO
import PIL.Image as Image


lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

with open('query/intents.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route('/get-solution', methods=['GET'])
def working():
    message = str(request.args['query']).lower()  # Input
    ints = predict_class(message)
    res = get_response(ints, intents)
    q = 0
    if ints == "Skin Disease":
        q = 1
    return {"message": str(res), "query": str(q)}


@app.route('/get-skin-disease', methods=['GET'])
def predict():
    SKIN_CLASSES = {
        0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
        1: 'Basal Cell Carcinoma',
        2: 'Benign Keratosis',
        3: 'Dermatofibroma',
        4: 'Melanoma',
        5: 'Melanocytic Nevi',
        6: 'Vascular skin lesion'
    }

    j_file = open('modelnew.json', 'r')
    loaded_json_model = j_file.read()
    j_file.close()
    model = model_from_json(loaded_json_model)
    model.load_weights('modelnew.h5')

    url = str(request.args['query'])
    response = requests.get(url)
    img1 = Image.open(BytesIO(response.content))

    def crop_center(pil_img, crop_width, crop_height):
        img_width, img_height = pil_img.size
        return pil_img.crop(((img_width - crop_width) // 2, (img_height - crop_height) // 2, (img_width + crop_width) // 2, (img_height + crop_height) // 2))

    size = min(img1.size[0], img1.size[1])
    img1 = crop_center(img1, size, size)
    img1 = img1.resize((224, 224), Image.ANTIALIAS)
    img1 = np.array(img1)
    img1 = img1.reshape((1, 224, 224, 3))
    img1 = img1 / 255.
    prediction = model.predict(img1)
    pred = np.argmax(prediction)
    disease = SKIN_CLASSES[pred]
    accuracy = prediction[0][pred]
    # print(f"You have symptoms of {disease}.")
    # print(f"Accuracy in this prediction was {int(accuracy*100)}%")
    return {"disease": str(disease), "accuracy": str(accuracy)}


if __name__ == "__main__":
    app.run()

