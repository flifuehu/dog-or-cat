# Data for this example is available at the Dogs vs. cats competition at Kaggle
# (https://www.kaggle.com/c/dogs-vs-cats/data). This example assumes the following
# files structure (needed by the ImageDataGenerator to work):
# - ../data/
#           train/
#                 cats/: 1000 cat images
#                 dogs/: 1000 dog images
#           test/
#                cats/: 200 cat images
#                dogs/: 200 dog images
#           validation/
#                      cats/: 200 cat images
#                      dogs/: 200 dog images

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt


def preprocess_input_vgg(x):
    """
    Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.

    Returns a numpy 3darray (the preprocessed image).
    """
    from keras.applications.vgg16 import preprocess_input
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]


batch_size = 16
nb_train_samples = 1000
nb_validation_samples = 400
epochs = 25

# load VGGNet with imagenet weights
vgg16 = VGG16(weights='imagenet')

fc2 = vgg16.get_layer('fc2').output
prediction = Dense(output_dim=1, activation='sigmoid', name='logit')(fc2)
model = Model(input=vgg16.input, output=prediction)
model.summary()

# freeze all but bottleneck layers for fine-tuning
for layer in model.layers:
    if layer.name in ['fc1', 'fc2', 'logit']:
        continue
    layer.trainable = False

# show which layers will be trained and which won't
layers_status = map(lambda l: '{}:\t{}'.format(l.name, l.trainable), model.layers)
print(*layers_status, sep='\n')

# compile with sgd optimizer and a small learning rate
sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['mae', 'acc'])

# set up data generators

# create feeder for train images
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(directory='../data/train',
                                                    target_size=[224, 224],
                                                    batch_size=batch_size,
                                                    class_mode='binary')

# create feeder for validation images
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)

validation_generator = validation_datagen.flow_from_directory(directory='../data/validation',
                                                              target_size=[224, 224],
                                                              batch_size=batch_size,
                                                              class_mode='binary')

# create feeder for test images
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)

test_generator = test_datagen.flow_from_directory(directory='../data/test',
                                                  target_size=[224, 224],
                                                  batch_size=batch_size,
                                                  class_mode='binary')

callbacks_array = [callbacks.EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=10, verbose=0, mode='max')]

# do fine-tuning
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=callbacks_array)

# Save weights of the trained model
model.save_weights('vgg16_dogs_vs_cats.h5')

# show some predictions
X_val, _ = next(test_generator)
y_pred = model.predict(X_val)

nb_sample = 16  # maximum "batch_size" samples
for x, y in zip(X_val[:nb_sample], y_pred.flatten()[:nb_sample]):
    img = array_to_img(x)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    # show the test image
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Cat or dog?')

    # show the predicted probabilities of each class
    axes[1].bar([0, 1], [1 - y, y])
    axes[1].set_title('Predicted class')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Cat', 'Dog'])

    plt.show()

print('Execution ended.')
