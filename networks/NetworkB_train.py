#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

 
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
size=224
import keras
from keras.models import Sequential, load_model
from keras.optimizers import SGD, RMSprop
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.activations import relu
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
K.set_image_dim_ordering('th')
import datetime
import math
import glob

def step_decay(epoch):
    initial_lrate = float(0.001)
    drop = float(0.3)
    epochs_drop = float(10.0)
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    lrate32=np.float32(lrate)
    return float(lrate32)




    
    
traindatagen = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.9,1.1],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1.0/255)


validdatagen = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        fill_mode='nearest',
        rescale=1.0/255)
meandatagen = ImageDataGenerator(
        fill_mode='nearest',
        rescale=1.0/255)

#
#import glob
#adresy=[]
#for i in range(5):
#    adres=glob.glob('fold'+str(i)+'*')
#    adresy.append(adres)
#adres=[]
#for element in adresy:
#    adres.append(element[-1]) 


#%%
for ess in range(1,7):
    for zbior in range(5):
    
        meangenerator=meandatagen.flow_from_directory('../data/'+str(zbior)+'/valid',class_mode='binary',classes=['ben','mal'],target_size=(size,size),batch_size=200,shuffle=False)
        for data in meangenerator:
            break
        data=data[0]
        traindatagen.fit(data)
        validdatagen.fit(data)
               
        traingenerator=traindatagen.flow_from_directory('../data/'+str(zbior)+'/train',class_mode='binary',classes=['ben','mal'],target_size=(size,size),batch_size=16,shuffle=True)
        validgenerator=validdatagen.flow_from_directory('../data/'+str(zbior)+'/valid',class_mode='binary',classes=['ben','mal'],target_size=(size,size),batch_size=16,shuffle=False)
        
        reg=0.0000005
        leak=0.2
        #
        model=Sequential()
        model.add(Convolution2D(64,3,3,border_mode='same' , init='glorot_normal', input_shape=(3,size,size)))
        
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        model.add(Convolution2D(128,3,3, border_mode='same' , init='glorot_normal'))    
       
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Convolution2D(256,3,3, border_mode='same' , init='glorot_normal'))    
       
        model.add(Activation('relu'))
        
        
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        
        model.add(Convolution2D(512,3,3, border_mode='same' , init='glorot_normal'))    
        
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
                                                                          
        
        model.add(Convolution2D(512,3,3, border_mode='same' , init='glorot_normal'))    
        
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024,init='glorot_normal'))    
        
        model.add(Activation('relu'))
        
        
        model.add(Dense(1024,init='glorot_normal'))    
     
        model.add(Activation('relu'))
        
        model.add(Dense(1,init='glorot_normal'))
        
        model.add(Activation('sigmoid'))
        model.summary()
         
        #model=load_model(adres[zbior])
        #print('Wczytano: '+adres[zbior])
        
        #%%
        
        
        class rysuj(keras.callbacks.Callback):
            
            def on_epoch_end(self, epoch, logs={}):
                czas=datetime.datetime.time(datetime.datetime.now())
                string='Epoch: '+str(epoch)+' '+str(logs['val_loss'])+' '+str(logs['val_acc'])+' '+str(czas)+'\n'
                text_file=open('../models/Network B/ess'+str(ess)+'/zbior'+str(zbior)+'_uczenie.txt','a')
                text_file.write(string)
                text_file.close()
        wykres=rysuj()
        
        from keras.callbacks import ModelCheckpoint
        from keras.callbacks import ReduceLROnPlateau
        
        filepath="../models/Network B/ess"+str(ess)+"/fold"+str(zbior)+"_epoch_"+"{epoch:02d}"+"_val_loss_"+"{val_loss:.4f}"+"_val_acc_"+"{val_acc:.3f}"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
        opt=SGD(lr=0.01,momentum=0.9, nesterov=True)
        reduce_lr=ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1, mode='auto', epsilon=0.00001, cooldown=0)
        model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
        model.fit_generator(traingenerator,samples_per_epoch=24563,nb_epoch=5000,validation_data=validgenerator,
                nb_val_samples=200,callbacks=[wykres,checkpoint,early_stop,reduce_lr])
        K.clear_session()

#%%




    
