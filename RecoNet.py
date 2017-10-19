
# coding: utf-8

# # RecoNet: Using Neural Networks to Reconstruct Phaseless Spectrograms
# This notebook is a first attempt at implementing a reconstructing neural network for recovering audio sounds from magnitude spectrograms only. This is all in hopes of building generative models for audio sounds.

# In[1]:

#get_ipython().magic(u'matplotlib inline')
#import IPython.display


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import gzip
import cPickle as pickle
import seaborn as sns
import random
import librosa
import sklearn

# Keras Imports
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers import Flatten, Reshape, TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from keras import backend as K
from keras.callbacks import TensorBoard

from librosa.display import waveplot, specshow
sns.set(style='ticks')
# seaborn.set(style='white')


# ## Loading Data

# In[2]:

# Loaded Spoken Digits Dataset
dbfile ='../SpokenDigitDB.pkl.gz'
with gzip.open(dbfile, 'rb') as ifile:
    df = pickle.load(ifile)
    print('File loaded as '+ dbfile)

df.head(n=3)    


# In[3]:

# Distribution of Audio Duration
a = [np.shape(x)[1] for x in df.Magnitude]
b = [np.shape(x) for x in df.Wave]

plt.figure(figsize=(6,4))
plt.subplot(211)
sns.boxplot(a)

plt.subplot(212)
sns.boxplot(b)
plt.tight_layout()


# In[4]:

# y = df.Wave[404][:5120]
# s = librosa.stft(y,n_fft=128-1,hop_length=64)
# s.shape  #(64, 80)


# In[5]:

# Padding & Truncating
smax = 80    # Max number of frames in STFT
wmax = 5120  # Corresponding max number of samples
spad = lambda a, n: a[:,0: n] if a.shape[1] > n else np.hstack((a, np.min(a[:])*np.ones([a.shape[0],n - a.shape[1]])))
wpad = lambda a, n: a[:n] if a.shape[0] > n else np.append(a,np.zeros(n-a.shape[0]))

df.Magnitude = df.Magnitude.apply(spad,args=(smax,))  # MaxLen Truncation Voodoo :D
df.Wave      = df.Wave.apply(wpad,args=(wmax,))

print(np.unique([np.shape(x)[1] for x in df.Magnitude]))
print(np.unique([np.shape(x)[0] for x in df.Wave]))


# In[6]:

# Plot K Random Examples
k  = 5
sr = 8000

sidx = random.sample(range(len(df)),k)
sidx = np.append(sidx,sidx)    

sns.set(style='white')
for i,j in enumerate(sidx):
    if i<k:
        plt.subplot(3,k,i+1)
        waveplot(df.Wave[j],sr=sr)
        plt.title('Digit:{1}'.format(j,df.Class[j]))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().get_xaxis().set_visible(False)

    else:
        plt.subplot(3,k,i+1)
        specshow(df.Magnitude[j],sr=sr)
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])


# In[7]:

# Play back an example!
#j = sidx[1]
#IPython.display.Audio(data=df.Wave[j], rate=sr)


# ## Data Prep

# In[8]:

# Prepare Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_validation import train_test_split

# Randomize DataFrame
df = df.sample(frac=1,random_state=32)

# Train Scaler
x_data = df.Magnitude.values
normsc = np.hstack(x_data)
scaler = MinMaxScaler().fit(normsc.T)

# Transform Data using Scaler
x_data = [scaler.transform(arr.T).T for arr in df.Magnitude.values]
x_data = np.dstack(x_data).transpose(2,1,0)


# Target Data (Decoder)
y_data = df.Wave.values             # Get waveform vectors
y_data = np.vstack(y_data)          # Stack to obtain 2D array of waveforms
ndim   = (y_data.shape[0],80,-1)    # Reshape into Windows of length 80
y_data = y_data.reshape(ndim)       # Use .Flatten() to recover

# Decoder Input Data (same as target, with offset 1 in time)
d_data = np.zeros_like(y_data)      
d_data[:,0, :] = 1.                 # Start Token: [1,1,1...1]
d_data[:,1:,:] = y_data[:,:-1,:]    # Decoder Input = Target plus 1 timestep: t + 1


# # Shuffle & Split
x_train,x_test,y_train,y_test,d_train,d_test=train_test_split(
    x_data,y_data,d_data,test_size=0.1, random_state=32)

# Print Dimensions
print 'Training Feature size:', x_train.shape
print 'Training D-Input size:', d_train.shape
print 'Training Target  size:', y_train.shape
print ''
print 'Testing  Feature size:', x_test.shape
print 'Testing  D-Input size:', d_test.shape
print 'Testing  Target  size:', y_test.shape


# In[9]:

# Testing Decoder Target & Input
plt.subplot(211)
plt.title('D-Input')
plt.plot(d_data[0].flatten())

plt.subplot(212)
plt.title('D-Target')
plt.plot(y_data[0].flatten())


# ## Sequence-To-Sequence Model

# In[10]:

# Parameters
frms,fbns  = x_train.shape[1:]   # FFT bins and frames in spectrogram
specg_size = (frms,fbns)         # Input spectrogram size
nspf       = y_train.shape[2]    # Number of samples per frame in output

E1  = 128
E2  = 80
D1  = E2
D2  = 100


# ### Model

# In[11]:

# Encoder
x      = Input(shape=specg_size,name='E-Input')
e1     = GRU(E1,activation='',return_sequences=True,name='E1')(x)
encode = GRU(E2,return_state='True',name='E2')
_,este = encode(e1)       # discard encoder output, use only state


# In[12]:

# Decoder
d      = Input(shape=(None,nspf),name='D-Input')
decode = GRU(D1,return_sequences=True,return_state=True,name='D1')
d1,_   = decode(d,initial_state=este)
d2     = GRU(D2,return_sequences=True,name='D2')(d1)
# dout   = TimeDistributed
dout   = GRU(nspf,return_sequences=True,name='Output')
y      = dout(d2)


# In[13]:

model = Model([x,d],y)
model.summary()
model.compile(optimizer='adam',
                loss='mean_squared_error')


# In[14]:

# Train Model
log = model.fit([x_train, d_train], y_train,
              batch_size=32,
              epochs=10,
              validation_split=0.2,
              initial_epoch=10)


# In[21]:

# Training Curves
plt.plot(log.epoch,log.history['loss'])
plt.plot(log.epoch,log.history['val_loss'],'g')
plt.title('Training Curves')
plt.xlabel('Epochs')
plt.xlabel('MSE Loss')
plt.legend(['Train','Valid'])


# In[ ]:



