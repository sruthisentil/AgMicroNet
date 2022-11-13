import os
import pickle
from tensorflow import keras
import importlib
import time
from keras.models import Model
IMAGE_SIZE = [320, 320]
import numpy as np
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#import external models
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50, Xception, EfficientNetB0, EfficientNetV2M, MobileNetV2

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, required=True)
parser.add_argument('--num_classes', type=int, required=True)
parser.add_argument('--type_of_crop', type=str, required=True)
parser.add_argument('--external_model', type=str, choices=['VGG16', 'ResNet50', 'Xception', 'EfficientNetB0', 'EfficientNetV2M', 'MobileNetV2'], help='External Modeling Approach', required=True)
parser.add_argument('--trainable_layers', type=str, choices=["fine_tune", "transfer_learn"], required=True)
args = parser.parse_args()

if args.trainable_layers == "fine_tune":
    trained = False
elif args.trainable_layers == "transfer_learn":
    trained = True

pickle_in = open('X_' + args.type_of_crop + '.pickle', "rb")
X = pickle.load(pickle_in)
print(X.shape)


pickle_in = open('y_' + args.type_of_crop + '.pickle', "rb")
y = pickle.load(pickle_in)

model_name = args.type_of_crop + "_" +  args.external_model + "_" + args.trainable_layers

# Define the K-fold Cross Validator
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True)

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

fold_no = 1

#callbacks
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# K-fold Cross Validation Model Training
for train_index, test_index in kfold.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=135,
        zoom_range=[0.75, 1.25],
        width_shift_range=[-5, 5],
        height_shift_range=0.1,
        shear_range=15,
        horizontal_flip=True,
        brightness_range=[0.3, 1.2],
        fill_mode="nearest")

    datagen.fit(X_train)
    datagen.fit(X_test)
    module = getattr(importlib.import_module("tensorflow.keras.applications"), args.external_model)
    base_model = module(weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in base_model.layers:
        layer.trainable = args.trainable_layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(500, activation='relu')(x)
    predictions = Dense(args.num_classes, activation='softmax')(x)

    # create a model object
    model = Model(inputs=base_model.input, outputs=predictions)

    # tell the model what cost and optimization method to use
    model.compile(
        loss='categorical_crossentropy',
        optimizer='RMSprop',
        metrics=['accuracy'])
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Training with callbacks
    from keras import callbacks
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='min')
    csv_filename = model_name + '.csv'
    csv_log = callbacks.CSVLogger(csv_filename, separator=',', append=True)

    hdf5_filepath = 'Best-weights-'+ model_name + '-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5'

    le = LabelEncoder()
    time_callback = TimeHistory()
    y = le.fit_transform(y)
    y_categorical= to_categorical(y, args.num_classes)
    y_train_categorical = to_categorical(y_train, args.num_classes)
    y_test_categorical = to_categorical(y_test, args.num_classes)
    # Fit data to model
    history = model.fit(X_train, y_train_categorical, batch_size=32, epochs=100, verbose=1, validation_data=(X_test, y_test_categorical),callbacks=[early_stopping,csv_log, time_callback])
    # save history
    np.save(model_name+ f'HISTORY_FOLD{fold_no}.npy', history.history)
    times = time_callback.times

    times_array = np.array(times)
    np.savetxt(model_name+ f"time_fold{fold_no}.csv", times_array, delimiter=",")
    # Generate generalization metrics
    scores = model.evaluate(X[test_index], y_categorical[test_index], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# Provide average scores
text_file = open("Metrics_" + model_name + ".txt", "w")
text_file.write('------------------------------------------------------------------------')
text_file.write('\nScore per fold')
for i in range(0, len(acc_per_fold)):
  text_file.write('\n------------------------------------------------------------------------')
  text_file.write( f'\n> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
text_file.write('\n------------------------------------------------------------------------')
text_file.write('\nAverage scores for all folds:')
text_file.write(f'\n> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
text_file.write(f'\n> Loss: {np.mean(loss_per_fold)}')
text_file.write('\n------------------------------------------------------------------------')
text_file.close()
saved_Model_Name = model_name.format(int(time.time()))
np.save(model_name + '_HISTORY'+ '.npy', history.history)

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("{name}.json".format(name=saved_Model_Name), "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("{name}.h5".format(name=saved_Model_Name))
print("Saved model to disk")

# load json and create model
json_file = open('{name}.json'.format(name=saved_Model_Name), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("{name}.h5".format(name=saved_Model_Name))
print("Loaded model from disk")
model.save('{name}.hdf5'.format(name=saved_Model_Name))
loaded_model = load_model('{name}.hdf5'.format(name=saved_Model_Name))
