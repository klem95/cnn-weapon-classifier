from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import keras

# dimensions of our images.
img_width, img_height = 150, 150

# Paths
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
prediction_data_dir = 'prediction_images'

# Gets the total no. of classes
classes = ImageDataGenerator().flow_from_directory(train_data_dir).class_indices

# Defining the total amount of samples in both the training and validation set
nb_train_samples = 770
nb_validation_samples = 1000

# Model attributes
epochs = 100
batch_size = 12 # The batch size represents the total amount of pictures that are included in each iteration.


best_model = keras.callbacks.ModelCheckpoint('custom_w_supervision_try' + '.h5', monitor='val_acc',save_best_only=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.25, patience=20,min_lr=0.000005)

def compileModel():
    print("compiling model")

    # Insureing that the images are in the correct format.
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # Defining the architecture of the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5)))
    # model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # The classifying part of the model
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer= regularizers.l2(0.1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) # This dropout insures that 50% of the connections are closed each time, in order to counter overfitting.
    model.add(Dense(4, kernel_regularizer= regularizers.l2(0.1))) # the model is closed off with a 4 dense layer, corresponding to the 4 different classes.
    model.add(Activation('softmax')) #



    model.compile(loss='categorical_crossentropy',optimizer= optimizers.adam(lr=1e-4),metrics=['accuracy']
                  )
    return model


def trainModel(model):

    # This augments the data. This is usefull when working with a small sample size
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    print("train generator")
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    print("validation generator")
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    print("starting ")
    hist = model.fit_generator(
        (train_generator),
        steps_per_epoch=nb_train_samples // batch_size, # The accumulated amount of steps
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=50
    )

    plotVal_plotLoss(hist)
    model.save_weights('custom_w_supervision_try.h5') # Saving the compile weights


# This function c
def predictImg(path, model):
   # model.load_weights('second_try.h5')
    imagep = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(imagep)
    x = x / 255  # Insures that images are normalized, so it can be compared test on a model that also used normalized training and validation images
    #x = preprocess_input(x)
    x = np.expand_dims(x, axis=0) # flattens the image
    prediction = model.predict(x) # Extract the prediction made by the model
    print(path)
    print(prediction)
    findLabel(prediction, 0.2, path)


def findLabel(test, threshold, path):
    if (max(test[0]) < threshold):
        print("no class could be defined for " + path + " with threshold 0.85")
    else:
        m = max(test[0])
        index = [i for i, j in enumerate(list(test[0])) if j == m]
        labeler(index[0], path)

def labeler(inp, pathname):
    label = list(classes.keys())[inp]
    print("The image '" + pathname + "' belongs to class: " + label) # Prints the prediction
    return 0

# This function generates graphs of the loss and the accuracy of the model
def plotVal_plotLoss (model) :

    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_plot_4 (simulated vgg1)')
    plt.show()

    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_plot_4 (simulated vgg1)')
    plt.show()


trainModel(compileModel())

#np.set_printoptions(suppress=True)
#model = compileModel()
np.set_printoptions(suppress=True, precision=3)
model = compileModel()
model.load_weights('custom_w_supervision_try.h5')

#predictImg(prediction_data_dir + '/hud.PNG', model) #  ansigt
predictImg(prediction_data_dir + '/ikke-gun.jpg', model) # ansigter


predictImg(prediction_data_dir + '/59kspz.jpg', model) # Knife
predictImg(prediction_data_dir + '/knive_m_hand.png', model) # knive

predictImg(prediction_data_dir + '/gun_m_hand.jpg', model) # Gun
predictImg(prediction_data_dir + '/gun_u_hand.jpg', model) # Gun


predictImg(prediction_data_dir + '/rifleman.jpg', model) # Rifle
predictImg(prediction_data_dir + '/rifle.jpeg', model) # Rifle




