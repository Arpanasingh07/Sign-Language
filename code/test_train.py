#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import itertools
import random
import warnings
import numpy as np
import cv2
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
warnings.simplefilter(action='ignore', category=FutureWarning)
from matplotlib import pyplot as plt
from tensorflow.keras.regularizers import l2













# In[2]:


train_path = r'C:\Users\pc\anaconda3\Sign_Language_Project Numbers\DATASET\Numbers dataset\TrainNewData'
test_path = r'C:\Users\pc\anaconda3\Sign_Language_Project Numbers\DATASET\Numbers dataset\TestNewData'

train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = train_path,target_size =(64,64),class_mode ='categorical',batch_size=8,shuffle=True)

test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = test_path,target_size =(64,64),class_mode ='categorical',batch_size=8,shuffle=True)

train_batches.class_indices
#print(len(train_batches))


# In[3]:


imgs, labels=next(train_batches)

#Plotting the images...
def plotImages(images_arr):
    fig,axes = plt.subplots(1,8, figsize =(30,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr,axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
plotImages(imgs)
print(imgs.shape)
print(labels)
print(len(train_batches))
print(len(imgs))


# In[21]:


model = Sequential()


model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding ='same'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))


model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

model.add(Flatten())


model.add(Dense(64,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10,kernel_regularizer = tf.keras.regularizers.l2(0.01),activation = 'softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=1,min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0,mode='auto')


#model.compile(optimizer=SGD(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
#reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=1,min_lr=0.0005)
#early_stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0,mode='auto')

#model.compile(optimizer=RMSprop(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])



# In[22]:


history2 = model.fit(train_batches, epochs=10, validation_data = test_batches)#, checkpoint])
imgs, labels = next(train_batches) #for getting next batch of imgs...

imgs, labels = next(test_batches)#for getting next batch of imgs...
scores = model.evaluate(imgs,labels,verbose=0)
print(f'{model.metrics_names[0]}of {scores[0]};{model.metrics_names[1]}of{scores[1]*100}%')


#model.save('best_model_dataflair.h5')
model.save('test_numbers_dataflair3.h5')

print(history2.history)

imgs,labels = next(test_batches)

model = keras.models.load_model(r"test_numbers_dataflair3.h5")

scores = model.evaluate(imgs,labels,verbose=0)
print(f'{model.metrics_names[0]}of{scores[0]};{model.metrics_names[1]}of{scores[1]*100}%')

model.summary()

scores #[loss,accuracy] on test data...
model.metrics_names


#word_dict = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',25:'z'}
word_dict = {0:'ZERO',1:'ONE',2:'TWO',3:'THREE',4:'FOUR',5:'FIVE',6:'SIX',7:'SEVEN',8:'EIGHT',9:'NINE'}
#word_dict = {0:'क',1:'ख',2:'ग',3:'घ',4:'च',5:'छ',6:'ज',7:'झ',8:'ट',9:'ठ',10:'ड',11:'ढ',12:'ण',13:'थ'}
predictions = model.predict(imgs,verbose=0)
print("predictions on a small set of test data--")
print("")
for ind, i in enumerate(predictions):
    print(word_dict[np.argmax(i)], end='   ')

plotImages(imgs)
print('Actual labels')
for i in labels:
    print(word_dict[np.argmax(i)],end='    ')
    
print(imgs.shape)


history2.history


# In[25]:


train_loss = history2.history['loss']
val_loss = history2.history['val_loss']
train_acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
xc = range(10)

plt.figure(1,figsize=(9,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('RMSprop-batch_size=8')
plt.grid(True)
plt.legend(['loss','validation loss'],loc='upper left')
#print(plt.style.available)
plt.style.use(['classic'])


plt.figure(2, figsize=(9,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.axis(ymin=0.0,ymax=1.5)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('RMSprop-batch_size=8')
plt.grid(True)
plt.legend(['Accuracy','validation Accuracy'],loc='lower right')
plt.style.use(['classic'])


# In[ ]:


#from PIL import Image
model = keras.models.load_model('test_numbers_dataflair3.h5')
img_path = r'C:\Users\pc\anaconda3\Sign_Language_Project Numbers\2.jpg'
#img = Image.open(img_path)
img = cv2.imread("number01.jpg")
img_x = 64
img_y = 64
img = cv2.resize(img,(img_x,img_y))

img = np.array(img, dtype=np.float32)
img = np.reshape(img, (-1, img_x, img_y, 3))
pred_prob = model.predict(img)
#pred_class = list(pred_prob).index(max(pred_prob))
#print(max(pred_prob), pred_class)
word_dict = {0:'ZERO',1:'ONE',2:'TWO',3:'THREE',4:'FOUR',5:'FIVE',6:'SIX',7:'SEVEN',8:'EIGHT',9:'NINE'}
output = word_dict[np.argmax(pred_prob)]
output

