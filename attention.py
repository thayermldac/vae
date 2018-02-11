# Define Frame Skipping Layer
def FrameSkipper(x):
    return x[:,0::2,:]    

def FrameSkipper_output_shape(input_shape):
    ishape = list(input_shape)
    ishape[-2] = np.arange(0,ishape[-2],2).shape[0]
    return tuple(ishape)


def Attention(x):
    hs = x[0]    # source hidden state - Encoder layer 
    ht = x[1]    # target hidden state - Decoder layer

    # et = tf.matmul(ht,hs,transpose_a=True) # score(ht,hs)
    # # et = K.dot(ht,hs)
    # et = K.tanh(et)
    # at = K.exp(et)
    # at /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())   
    _,source_sequence_length,num_units = K.int_shape(x) 
    # Create an attention mechanism
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, x,memory_sequence_length=source_sequence_length)









# Sample
x = Input(shape=(16,64),name='Input')
e = GRU(10,return_sequences=True,name='Encoder')(x)
d = GRU(20,return_sequences=True,name='Decoder')(e)
f = Flatten()(d)
y = Dense(1,activation='sigmoid',name='Output')(f)

model = Model(x,y)
model.summary()
model.compile(optimizer='sgd',loss='binary_crossentropy')



d._keras_shape

K.matmul(e,d)
tf.matmul(e,d)

#----- 
a  = K.placeholder(shape=(None,2, 3))
b  = K.placeholder(shape=(None,3, 4))
ab = K.dot(a, b)
K.int_shape(ab)

#-----
a  = K.placeholder(shape=(32, 28, 3))
b  = K.placeholder(shape=(3, 4))
ab = K.dot(a, b)
K.int_shape(ab)


#-----
a  = K.placeholder(shape=(None,16,10))
b  = K.placeholder(shape=(None,16,20))
ab = K.dot(a, b)
K.int_shape(ab)


