# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 02:10:47 2020

@author: mukmc
"""

import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model = load_model('Pretrained_Models/facenet_keras.h5')

i=0
for l in model.layers:
    if(i>420):
        break
    else:
        model.layers[i].trainable=False
    i=i+1

newmod=Sequential()
newmod.add(model)
newmod.add(Dense(units=3,activation='softmax'))
newmod.compile(optimizer = 'adam', loss = 'categorical_crossentropy')



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False)
training_set = train_datagen.flow_from_directory(
        'actual_data/train',
        target_size=(160, 160),
        batch_size=32,shuffle=True,
        class_mode='categorical')

val_set = val_datagen.flow_from_directory(
        'actual_data/val',
        target_size=(160,160),
        batch_size=32,shuffle=True,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'actual_data/test',
    target_size=(160,160),
    batch_size=32,shuffle=False,
    class_mode='categorical')


checkpoint = ModelCheckpoint("best_mod.h5", monitor='val_loss', verbose=1,save_best_only=True,mode='min')


history=newmod.fit_generator(
        training_set,
        steps_per_epoch=(3256/32),
        epochs=20,
        validation_data=val_set,
        validation_steps=(406/32), callbacks=[checkpoint])


best_mod=load_model("best_mod.h5")


print(best_mod.evaluate_generator(test_set))

import numpy

predictions = best_mod.predict_generator(test_set)
# Get most likely class
predicted_classes = numpy.argmax(predictions, axis=1)

true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())  


import sklearn.metrics as metrics
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)


from sklearn.metrics import confusion_matrix

confusion_matrix(true_classes, predicted_classes)



real_generator = ImageDataGenerator(rescale=1./255,horizontal_flip=False)
real_set = real_generator.flow_from_directory(
    "real", # Put your path here
     target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

yopred = best_mod.predict_generator(real_set)
yoclass = numpy.argmax(yopred, axis=1)

