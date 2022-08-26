
import nltk
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, request
import nltk

lemmatizer=WordNetLemmatizer()
app = Flask(__name__)

with open('query/intents.json') as json_file:
  intents = json.load(json_file)

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
  sentence_words=nltk.word_tokenize(sentence)
  sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
  return sentence_words

def bag_of_words(sentence):
  sentence_words=clean_up_sentence(sentence)
  bag=[0]*len(words)
  for w in sentence_words:
    for i,word in enumerate(words):
      if word == w:
        bag[i]=1
  return np.array(bag)

def predict_class(sentence):
  bow=bag_of_words(sentence)
  res=model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD=0.25
  results=[[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]

  results.sort(key=lambda x:x[1],reverse=True)
  return_list=[]
  for r in results:
    return_list.append({'intent': classes[r[0]],'probability':str(r[1])})
  return return_list

def get_response(intents_list,intents_json):
  tag=intents_list[0]['intent']
  list_of_intents=intents_json['intents']
  for i in list_of_intents:
    if i['tag']==tag:
      result=random.choice(i['responses'])
      break
  return result


@app.route('/get-solution', methods=['GET'])
def working():
  message = str(request.args['query'])  # Input
  ints=predict_class(message)
  res=get_response(ints,intents)
  return {"message" : res}


if __name__ == "__main__":
    app.run()
# =======
# from distutils.log import ERROR
# import nltk
# import random
# import numpy as np
# import json
# import pickle
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
# lemmatizer=WordNetLemmatizer()

# with open('intents.json') as json_file:
#   intents = json.load(json_file)

# words=pickle.load(open('words.pkl','rb'))
# classes=pickle.load(open('classes.pkl','rb'))
# model=load_model('chatbotmodel.h5')

# def clean_up_sentence(sentence):
#   sentence_words=nltk.word_tokenize(sentence)
#   sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
#   return sentence_words

# def bag_of_words(sentence):
#   sentence_words=clean_up_sentence(sentence)
#   bag=[0]*len(words)
#   for w in sentence_words:
#     for i,word in enumerate(words):
#       if word == w:
#         bag[i]=1
#   return np.array(bag)

# def predict_class(sentence):
#   bow=bag_of_words(sentence)
#   res=model.predict(np.array([bow]))[0]

#   ERROR_THRESHOLD=0.25
#   results=[[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]

#   results.sort(key=lambda x:x[1],reverse=True)
#   return_list=[]
#   for r in results:
#     return_list.append({'intent': classes[r[0]],'probability':str(r[1])})
  
#   return return_list

# def get_response(intents_list,intents_json):
#   tag=intents_list[0]['intent']
#   list_of_intents=intents_json['intents']
#   for i in list_of_intents:
#     if i['tag']==tag:
#       result=random.choice(i['responses'])
#       break
#   return result

# print("Start")

# def working():
#   message=str.lower(input("")) #Input
#   ints=predict_class(message)
#   res=get_response(ints,intents)
#   return res

# while True:
#   print(working())
# >>>>>>> 120429c3dabaa59f25429cc2292b6de9e40f96c3:chatbot_py.py
