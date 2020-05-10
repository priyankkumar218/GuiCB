import nltk
# nltk.download('punkt')
import tensorflow
import tflearn
import pyttsx3
import webbrowser
from tkinter import simpledialog
import tkinter as tk
try:
    import ttk as ttk
    import ScrolledText
except ImportError:
    import tkinter.ttk as ttk
    import tkinter.scrolledtext as ScrolledText
import time

engine = pyttsx3.init()

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import random
import json
import pickle

with open('intents.json') as file:
    data = json.load(file)

# print(data["intents"])
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in  words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_Of_Words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)



class TkinterGUIExample(tk.Tk):

    def __init__(self, *args, **kwargs):
        """
        Create & set window variables.
        """
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("Covpan")

        self.initialize()

    def initialize(self):
        """
        Set window layout.
        """
        self.grid()

        self.user_name = simpledialog.askstring("Covpan", prompt="What's your Name?")

        self.welcome_user_lbl = tk.Label(self, text='Welcome '+ self.user_name, font=1)
        self.welcome_user_lbl.grid(column=1, row=1)

        self.conversation_lbl = tk.Label(self, anchor=tk.E, text='Conversation:', font=1)
        self.conversation_lbl.grid(column=0, row=2, sticky='nesw', padx=3, pady=3)

        self.conversation = ScrolledText.ScrolledText(self, state='disabled')
        self.conversation.grid(column=0, row=3, columnspan=2, sticky='nesw', padx=3, pady=3)

        self.usr_input = tk.Entry(self, state='normal', font=1)
        self.usr_input.grid(column=0, row=4, sticky='nesw', padx=3, pady=3)

        self.respond = tk.Button(self, text='Send', command=self.get_response, font=1)
        self.respond.grid(column=1, row=4, sticky='nesw', padx=3, pady=3)

    def get_response(self):
        """
        Get a response from the chatbot and display it.
        """
        user_input = self.usr_input.get()
        if(user_input == "quit"): exit()
        else:
            results = model.predict([bag_Of_Words(user_input, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            # print(results)
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg['responses']
            speechvar = random.choice(responses)
            if("volunteer" in user_input):
                speak("Opening browser")
                webbrowser.open_new("https://covidwarriors.gov.in/")
            # print(speechvar)
            # speak(speechvar)
            self.conversation['state'] = 'normal'
            self.conversation.insert(tk.END,  self.user_name + ": " + user_input + "\n" + "Covpan: " + str(speechvar) + "\n")
            self.usr_input.delete(0, tk.END)
            self.conversation['state'] = 'disabled'
            time.sleep(0.5)

gui_example = TkinterGUIExample()
gui_example.mainloop()