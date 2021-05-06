from nltk.stem.lancaster import LancasterStemmer
import pandas as pd
import json
import pickle
import nltk
import numpy
import tflearn
import random
from flask import Flask, jsonify, request, render_template_string
import time

app = Flask(__name__)
app.secret_key = 'super_secret_key'

#   Just some space

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load('model.tflearn')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

# Just another space


@app.route('/')
def index():
    return render_template_string("Hello")


@app.route("/bot", methods=["POST"])
def response():
    query = dict(request.form)['query']
    results = model.predict([bag_of_words(query, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.7:
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
            elif tag == 'goodbye':
                result = random.choice(responses)
        result = random.choice(responses)
    else:
        result = 'I didn\'t get that, please try once more'

    # query = dict(request.form)['query']
    # result = query + " " + time.ctime()
    # print(query + "*************")
    # print(result + "*************")
    return jsonify({"response": result})
    # return render_template_string('Hello')


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0",)
