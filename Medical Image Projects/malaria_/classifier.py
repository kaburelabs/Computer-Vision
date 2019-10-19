#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:57:25 2019

@author: leonardo
"""


# import the necessary packages
import matplotlib 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from pyimagesearch.resnet import ResNet
from pyimagesearch import config
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

matplotlib.use("Agg")


# argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


# defining the total number of epochs to train for along with the
# initial learning rate and batch size
NUM_EPOCHS = 50
INIT_LR = 1e-1
BS = 64
 
# function to decay the learning rate
def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0
 
	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 
	# return the new learning rate
	return alpha


# geting total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(rescale=1 / 255.0,
                              ation_range=20, m_range=0.05,	
                              width_shift_range=0.05, height_shift_range=0.05,	
                              shear_range=0.05,	horizontal_flip=True, fill_mode="nearest")
 
# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)


# training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)
 
# validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)
 
# testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)


# initialize the ResNet model and compile it (Deep convolution neural nets)
model = ResNet.build(64, 64, 3, # Image size and rgb multi-channel
                     2, # two classes
                     (3, 4, 6),	# Stacking levels
                     (64, # filters alone
                      128, 256, 512), # filters in stack
                     reg=0.0005) 
                     
# Defining the optimizer
opt = SGD(lr=INIT_LR, momentum=0.9) 

# compiling the NN structure with loss, opt and metric parameters
model.compile(loss="binary_crossentropy", optimizer=opt,	
              metrics=["accuracy"])


# set of callbacks
callbacks = [LearningRateScheduler(poly_decay)]

# fit the model
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS, # total runs to fit all dataset
	validation_data=valGen, # data to validation
	validation_steps=totalVal // BS,
	epochs=NUM_EPOCHS, # total epochs run
	callbacks=callbacks # to decay the learning rate after each epoch
    )


# reseting the testing generator and then use the trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,	steps=(totalTest // BS) + 1)
 
# for each image in the testing set find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
 
# classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))


# ploting the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


