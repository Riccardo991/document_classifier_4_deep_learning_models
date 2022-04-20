
# Document classifier 4 deep learning models 
#The dataset has two classes in a 1 to 4 ratio. 
# The purpose of this script is to implement and test different neural network models. 

import pandas as pd 
import numpy as np
import re, time, pickle 
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, log_loss, f1_score, balanced_accuracy_score, roc_auc_score 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential 
from keras.initializers import RandomUniform
from keras.layers import Dense, Dropout, Embedding, Flatten, SimpleRNN, LSTM, SpatialDropout1D, GRU, GlobalMaxPool1D, Conv1D,MaxPooling1D  
from keras import optimizers 
from keras.backend import clear_session

print("go ")
df = pd.read_excel('...\\corpus_tiket_11.xlsx')
print("df size ",df.shape)
print("labels ", Counter(df['targhet_class']))

#  vectorize the corpus of texts 
maxWords = 6000
lenText = 40
numEmbedding = 50
tk = Tokenizer (num_words= maxWords )
tk.fit_on_texts( df['corpus'].values)
dicLen = tk.word_index
print("the size of vocavolary is di ", len( dicLen))
x_set = tk.texts_to_sequences( df['corpus'].values)
x_set =  pad_sequences(x_set, maxlen= lenText, padding='post' )
print(" x_set ",x_set.shape)
y_set = df['targhet_class'].values
print(" y_set ",y_set.shape)

x_tr, x_ts, y_tr,  y_ts = train_test_split(x_set, y_set, test_size=0.1, random_state=37)
print("trainset size  ",x_tr.shape," testset size ",x_ts.shape)

clear_session()
t1 = time.time()

# define dense models 
m1 = Sequential()
m1.add( Embedding( input_dim=maxWords, output_dim=numEmbedding, input_length= lenText ))
m1.add(Flatten())
m1.add( Dense(64, activation='relu'))
m1.add( Dropout(0.25))
m1.add( Dense(16, activation='relu'))
m1.add( Dropout(0.25))
m1.add(Dense(1, activation='sigmoid'))

#print(m1.summary())
wc = {0:1, 1:4}
m1.compile(optimizer= 'sgd', loss='binary_crossentropy', metrics=['accuracy'])
m1.fit(x_tr, y_tr, epochs=100, class_weight = wc, batch_size=256, validation_split=0.1, verbose= True )

t2 = time.time()
print("  time training model 1 ",int(t2-t1)," seconds ")

# evaluate the model
lg1_tr, ac1_tr = m1.evaluate(x_tr, y_tr)
print("training m1: accuracy%.4f, loss %.4f " %(ac1_tr, lg1_tr))

y1_pred = m1.predict_classes(x_ts)
y1_ac = accuracy_score(y_ts, y1_pred)
y1_f1 = f1_score(y_ts, y1_pred, average='weighted' )
y1_bac = balanced_accuracy_score(y_ts, y1_pred)
y1_prob = m1.predict(x_ts)
y1_los = log_loss(y_ts, y1_prob)
y1_roc = roc_auc_score (y_ts, y1_prob,average='weighted' )
print("test m1: accuracy=%.4f, f1= %.4f, balanced accuracy=%.4f, loss=%.4f, roc=%.4f " %(y1_ac, y1_f1, y1_bac, y1_los, y1_roc))


t3 = time.time()

#  define a  conutional net 
m2 = Sequential()
m2.add( Embedding( input_dim=maxWords, output_dim=numEmbedding, input_length= lenText ))
m2.add( Conv1D(32, 3, activation='relu' ))
m2.add( MaxPooling1D(  pool_size=2))
m2.add( Dropout(0.3))
m2.add( Flatten())
m2.add( Dense(64, activation='relu'))
m2.add( Dropout(0.3))
m2.add( Dense(16, activation ='relu'))
m2.add( Dropout(0.3))
m2.add(Dense(1, activation='sigmoid'))

#print(m2.summary())
m2.compile(optimizer= 'sgd', loss='binary_crossentropy', metrics=['accuracy'])
m2.fit(x_tr, y_tr, epochs=100, class_weight = wc, batch_size=256, validation_split=0.1, verbose= True )

t4 = time.time()
print(" the training time for m2 is %.2f seconds " %(t4 - t3))
# evaluate the model 
ls2_tr, ac2_tr = m2.evaluate(x_tr, y_tr)
print("training m2: accuracy%.4f, loss %.4f " %(ac2_tr, ls2_tr))

y2_pred = m2.predict_classes(x_ts)
y2_ac = accuracy_score(y_ts, y2_pred)
y2_f1 = f1_score(y_ts, y2_pred, average='weighted' )
y2_bac = balanced_accuracy_score(y_ts, y2_pred)
y2_prob = m2.predict(x_ts)
y2_los = log_loss(y_ts, y2_prob)
y2_roc = roc_auc_score (y_ts, y2_prob,average='weighted' )
print("test m2: accuracy=%.4f, f1= %.4f, balanced accuracy=%.4f, loss=%.4f, roc=%.4f " %(y2_ac, y2_f1, y2_bac, y2_los, y2_roc))

t5 = time.time()

# define a LSTM net 
m3 = Sequential()
m3.add( Embedding( input_dim=maxWords, output_dim=numEmbedding, input_length= lenText ))
m3.add( LSTM(64,  dropout=0.2,recurrent_dropout=0.2)) 
m3.add( Dense(16, activation='relu'))
m3.add( Dropout(0.3))
m3.add(Dense(1, activation='sigmoid'))

#print(m3.summary())
m3.compile(optimizer= 'sgd', loss='binary_crossentropy', metrics=['accuracy'])
m3.fit(x_tr, y_tr, epochs=100, class_weight = wc, batch_size=256, validation_split=0.1, verbose=True )

t6 = time.time()
print(" the training time for m3 is %.2f seconds " %(t6 - t5))
# evaluate the model
ls3_tr, ac3_tr = m3.evaluate(x_tr, y_tr)
print("training m3: accuracy%.4f, loss %.4f " %(ac3_tr, ls3_tr))

y3_pred = m3.predict_classes(x_ts)
y3_ac = accuracy_score(y_ts, y3_pred)
y3_f1 = f1_score(y_ts, y3_pred, average='weighted' )
y3_bac = balanced_accuracy_score(y_ts, y3_pred)
y3_prob = m3.predict(x_ts)
y3_los = log_loss(y_ts, y3_prob)
y3_roc = roc_auc_score (y_ts, y3_prob,average='weighted' )
print("test m3: accuracy=%.4f, f1= %.4f, balanced accuracy=%.4f, loss=%.4f, roc=%.4f " %(y3_ac, y3_f1, y3_bac, y3_los, y3_roc))

# save the model
m3.save("LSTM_net.h5")

token = 'tokenizer_model.sav'
pickle.dump(tk, open(token, 'wb') )

print("end")