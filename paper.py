from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

BATCH_SIZE = 3
MAX_EPOCH = 1000
IMAGE_SIZE = (1024,1024)
TRAIN_IM = 455
VALIDATE_IM = 40

model = Sequential()
# model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal',
#                  input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(32, 5, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal'))

# model.add(Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(UpSampling2D(size=(2,2)))
# model.add(Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(UpSampling2D(size=(2, 2)))
# model.add(Conv2D(32, 5, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(UpSampling2D(size=(2, 2)))
# model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
# model.add(Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal'))

# print(model.summary())

def iou(y_true, y_pred):
    y_true = K.cast(K.greater(y_true, 0.5), dtype='float32')
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(K.clip(y_true + y_pred, 0, 1), axis=3), axis=2), axis=1)
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

model = load_model('re_model2.h5',custom_objects={'iou':iou})



optimizer = optimizers.SGD(lr=0.001,momentum=0.9, nesterov=False) 
model.compile(optimizer=  optimizer, loss='binary_crossentropy', metrics=['accuracy',iou])

def myGenerator(type):
    train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=180.,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

    input_generator = train_datagen.flow_from_directory(
        'textlocalize/'+type,
        classes = ['Input'],
        class_mode=None,
        color_mode='rgb',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    expected_output_generator =  train_datagen.flow_from_directory(
        'textlocalize/'+type,
        classes = ['Output'],
        class_mode=None,
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    while True:
        in_batch = input_generator.next()
        out_batch = expected_output_generator.next()
        yield in_batch, out_batch

checkpoint = ModelCheckpoint('re_model4.h5', verbose=1, monitor='val_iou',save_best_only=True, mode='max')
h = model.fit_generator(myGenerator('train'),
                        steps_per_epoch=TRAIN_IM/BATCH_SIZE,
                        epochs=MAX_EPOCH,
                        validation_data=myGenerator('validation'),
                        validation_steps=VALIDATE_IM/BATCH_SIZE,
                        callbacks=[checkpoint])

plt.plot(h.history['iou'])
plt.plot(h.history['val_iou'])
plt.show()