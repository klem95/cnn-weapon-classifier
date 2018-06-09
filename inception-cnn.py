from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras import applications
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# paths:
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
prediction_data_dir = 'prediction_images'

classes = ImageDataGenerator().flow_from_directory(train_data_dir).class_indices # Gets classes from folder structure

img_h, img_w = 150, 150
train_sample_size = 770 # Specify the amount of images in training samples
validation_sample_size = 400 # and validation samples

batch_size = 16 # amount of pictures that is trained at once

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_h, img_w)
else:
    input_shape = (img_h, img_w, 3)


def compileModel():
    model = applications.InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet')
    x = model.output
    x = Flatten()(x) # flattening output from
    # adding a fully connected layer to InceptionV3
    x = Dense(256, activation='relu')(x)
    # Connecting to a 4 class dense layer, using softmax for prediction
    predictions = Dense(4, activation='softmax')(x)

    train_model = Model(inputs=model.input, outputs=predictions) #creating a new model, with v3 as input and our dense as output

    for layer in model.layers:
        layer.trainable = False # here we disable training for the v3 network

    #Compiling the network using adam and learning rate as 0.0001
    train_model.compile(optimizer=optimizers.adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'] )
    return train_model


def train_model(input_model):
    epochs = 50
    batch_size = 16
    # this is the augmentation configuration we will use for training

    train_datagen = ImageDataGenerator( # image
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)


    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_w, img_h),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)


    print("validation generator")
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_w, img_h),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    print("starting ")
    hist = input_model.fit_generator(
        (train_generator),
        steps_per_epoch=train_sample_size // batch_size,
        epochs=epochs,
        validation_data= validation_generator,
        nb_val_samples = 50)

    plotVal_plotLoss(hist)
    input_model.save_weights('inceptionV3Weights.h5')

    #validation_steps=nb_validation_samples // batch_size)


def plotVal_plotLoss (model) :

    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def predictImg(path, model):
    imagep = image.load_img(path, target_size=(img_w, img_h))
    x = image.img_to_array(imagep)
    x = x / 255
    #x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    findLabel(prediction, 0.85, path)
    print(prediction)


def findLabel(test, threshold, path):
    if (max(test[0]) < threshold):
        print("no class could be defined for " + path + " with threshold 0.85")
    else:
        m = max(test[0])
        index = [i for i, j in enumerate(list(test[0])) if j == m]
        labeler(index[0], path)


def labeler(inp, pathname):
    label = list(classes.keys())[inp]
    print("The image '" + pathname + "' belongs to class: " + label)
    return 0


train_model(compileModel())

np.set_printoptions(suppress=True, precision=3)
model = compileModel()
#model.load_weights('inceptionV3Weights.h5')

predictImg(prediction_data_dir + '/hud.PNG', model) #  ansigt
predictImg(prediction_data_dir + '/ikke-gun.jpg', model) # ansigter
predictImg(prediction_data_dir + '/rifle.jpeg', model) # Rifle
predictImg(prediction_data_dir + '/59kspz.jpg', model) # Kniv

predictImg(prediction_data_dir + '/gun_m_hand.jpg', model) # Gun
predictImg(prediction_data_dir + '/gun_u_hand.jpg', model) # Gun

predictImg(prediction_data_dir + '/standart_knive.jpg', model) # knive
predictImg(prediction_data_dir + '/hunter_knive.jpg', model) # knive
predictImg(prediction_data_dir + '/knive_m_hand.png', model) # knive

predictImg(prediction_data_dir + '/ak_rifle.jpg', model) # Rifle
predictImg(prediction_data_dir + '/rifle_m_man.jpg', model) # Rifle

np.set_printoptions(suppress=False, precision=10)