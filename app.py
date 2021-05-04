from nltk.stem.lancaster import LancasterStemmer
import pandas as pd
import json
import pickle
import nltk
import numpy
import tflearn
# from tensorflow.python.framework import ops
# import tensorflow as tf
import random
from flask import Flask, jsonify, request, render_template_string
import time

app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string("Yes")

#   Just some space


# import sys
stemmer = LancasterStemmer()


with open("intents.json") as file:
    data = json.load(file)
    # data = pd.read_json(file)

try:
    with open("datad.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load('model.tflearn')
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


# def chat():
#     print("start talking with the bot (Type 'quit' to stop!")
#     while True:
#         inp = input("You: ")
#         if inp.lower() == "quit":
#             break
#         results = model.predict([bag_of_words(inp, words)])[0]
#         results_index = numpy.argmax(results)
#         tag = labels[results_index]
#         bot_response = random.choice(responses)

#         if results[results_index] > 0.7:
#             for tg in data['intents']:
#                 if tg['tag'] == tag:
#                     responses = tg['responses']
#                 elif tag == 'goodbye':
#                     # print('Bot: ', random.choice(responses))
#                     quit()
#             # print('Bot: ', random.choice(responses))
#         else:
#             # print('Bot: I didn\'t get that, please try once more')


# chat()


# Just another space

@app.route("/bot")
def response():
    query = dict(request.form)['query']
    results = model.predict([bag_of_words(query, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    bot_response = random.choice(responses)

    if results[results_index] > 0.7:
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
            elif tag == 'goodbye':
                result = random.choice(responses)
                # print('Bot: ', random.choice(responses))
                quit()
        result = random.choice(responses)
        # print('Bot: ', random.choice(responses))
    else:
        result = 'I didn\'t get that, please try once more'
        # print('Bot: I didn\'t get that, please try once more')

    # query = dict(request.form)['query']
    # result = query + " " + time.ctime()
    return jsonify({"response": result})
    # print(result + "*************")
    # return render_template_string('Hello')


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0",)
