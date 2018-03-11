import pandas as pd
import numpy as np
import operator
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle as pkl
with open('sent.pkl','rb') as handle:
    dataset=pkl.load(handle)
review_text=dataset[['overall','summary']]
max(len(w) for w in review_text['summary'].astype(np.str))
max(review_text['summary'],key=len)
review_text['summary']=review_text['summary'].apply(lambda x:x.lower())
review_text['summary']=review_text['summary'].apply((lambda x: re.sub('[^a-zA-z\s]','',x)))
for i in range(5):
    print('rating=',i+1,'\t',review_text[review_text['overall']==i+1].size)
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(review_text['summary'].values)
X = tokenizer.texts_to_sequences(review_text['summary'].values)
X = pad_sequences(X)
X_min=pad_sequences(X[:10000])

#model
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X_min.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(5,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y_min = pd.get_dummies(review_text['overall']).values[:10000]
X_train, X_test, Y_train, Y_test = train_test_split(X_min,Y_min, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 64
model.fit(X_train, Y_train, nb_epoch = 30, batch_size=batch_size, verbose = 2)

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

import ipywidgets as widgets
from IPython.display import  display

text=widgets.Text(value='it was awesome')
text.on_submit(predict_sentiments)
display(text)

def predict_sentiments(b):
    _X=tokenizer.texts_to_sequences([b])
    _X=pad_sequences(_X,maxlen=43)
#     print(_X)
    op_l=list(model.predict(_X).ravel())
    i,v=max(enumerate(op_l),key=operator.itemgetter(1))
    i+=1
    res='\n\n\npredicted rating is:\t'+str(i)+'\n\n'
    if i==1:
        sent='poor,very bad, Hated it'
    elif i==2:
        sent='not bad, i don\'t prefer it'
    elif i==3:
        sent='neutral, Can\'t say anything'
    elif i==4:
        sent='good, satisfactory'
    else:
        sent='Excellent, Loved it'
    res+='Sentiment might be:\t'+str(sent)
    return res

model.save('mk2.h5')

from tkinter import *

class Application(Frame):
    def __init__(self, master=None):
            master.title('Sentiment Analyser UI')
            master.geometry('500x300')
            Frame.__init__(self, master)
            self.pack()
            self.createWidgets()
    def createWidgets(self):
            self.Ftop=Frame(self,height='200')
            self.Ftop.pack(fill=None, expand=False)
            self.info_label=Label(self.Ftop,text='Enter a text to predict its emotion')
            self.info_label.pack(side=TOP)
            self.e = Entry(self.Ftop,width=80)
            self.e.pack()
            self.e.focus_set()
            self.b = Button(self.Ftop, text="Predict", width=10, command=self.callback)
            self.b.place(relx=0.5, rely=0.5, anchor=NW)
            self.b.pack()
            self.opVar=StringVar(value='Sentiment:')
            self.op=Label(self.Ftop,textvariable=self.opVar,text="Helvetica", font=("Helvetica", 16))
            self.op.pack(side=BOTTOM)
    def callback(self):
            res=predict_sentiments(self.e.get())
            self.opVar.set(res)


root = Tk()
app = Application(master=root)
app.mainloop()
