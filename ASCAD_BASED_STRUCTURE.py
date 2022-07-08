# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:25:57 2022

@author: Nex_Os
"""
## Import section
import os
import os.path
import sys
import h5py
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, BatchNormalization, Activation, Add, add
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2, l1
import matplotlib.pyplot as plt


## Helper data structures.

# \brief The AES SBox that we will use to compute the rank
AES_Sbox = np.array([
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ])

## Helper functions

# \brief Checks if there is the file under given path
#
# \param file_path path to the checked file.
def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1) 
    return

# \brief ASCAD helper to load profiling and attack data (traces and labels)
#        Loads the profiling and attack datasets from the ASCAD database
#
# \param ascad_database_file - path to the file to load.
# \param load_metadata - should the metadata(key, mask, etc.) be included into the set.
def load_ascad(ascad_database_file, load_metadata=False, DPA_format= False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file     = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
     
    # If classical format
    if ( DPA_format == False):
        # Load profiling traces
        X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
        # Load profiling labels
        Y_profiling = np.array(in_file['Profiling_traces/labels'])
        # Load attacking traces
        X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
        # Load attacking labels
        Y_attack = np.array(in_file['Attack_traces/labels'])
        if load_metadata == False:
            return (X_profiling, Y_profiling), (X_attack, Y_attack)
        else:
            return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])
    
    # If use DPA/DDLA format
    # Load profiling data
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.uint8)
    Y_profiling = np.array(in_file['Profiling_traces/metadata']['plaintext'], dtype=np.uint8)
    key_profiling = np.array(in_file['Profiling_traces/metadata']['key'], dtype=np.uint8) # exclusively for accuracy metrics
    #Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.uint8)
    Y_attack = np.array(in_file['Attack_traces/metadata']['plaintext'], dtype=np.uint8)
    key_attack = np.array(in_file['Attack_traces/metadata']['key'], dtype=np.uint8) # exclusively for accuracy metrics
    
    return (X_profiling, Y_profiling, key_profiling), (X_attack, Y_attack, key_attack)

# \brief Creates the labels for given plaintext-key pair
#
# \param plaintext_set - plaintexts.
# \param keybyte - key. TODO: make it also iterable, but only if array is provided
# \param keymask - the mask used to reduce the number of classes. MSB=0x80, LSB=0x01, Identity= 0xFF. TODO: add support for HW mapping
# \param keybyte - position of the attacked byte, the count starts from 0. For default ASCAD subset it is 2.
#
# \ret Array of output values of the first  
def create_labels(plaintext_set, keybyte , keyMask=0x1, keybytePos = 2):
    return (AES_Sbox[plaintext_set[:,keybytePos]^keybyte]&keyMask)

# \brief Creates the 4 subplots, allowing to depict the loss and accuracy in human-readable form
#
# \ret figure
# \ret array of subplots, with pre-defined titles
def formGraph():
    fig, art= plt.subplots(4, sharex=True)
    fig.tight_layout(pad=1.0)
    art[0].grid()
    art[0].minorticks_on()
    art[0].set_title("Training Accuracy",fontsize='small')
    art[1].grid()
    art[1].minorticks_on()
    art[1].set_title("Training Loss", fontsize='small')
    art[2].grid()
    art[2].minorticks_on()
    art[2].set_title("Validation Accuracy", fontsize='small')
    art[3].grid()
    art[3].minorticks_on()
    art[3].set_title("Validation Loss", fontsize='small')
    return fig,art
    
# \brief depicts the provided data on the plot
#
# \param art - array of 4 subplots. Fromed by function formGraph() per default.
# \param history - array of metric values, returned by trained model.
# \param key - key, used for labeling of data.
# \param rightKey - right key.
def plotStep(art, history, key, rightKey):
    zorder=key
    color="grey"
    if (key==rightKey):
        color="red"
        zorder=0x100
    art[0].plot(history.history['accuracy'],  label=key,color=color, zorder=zorder)
    art[1].plot(history.history['loss'], label=key,color=color,zorder=zorder)
    art[2].plot(history.history['val_accuracy'],  label=key,color=color, zorder=zorder)
    art[3].plot(history.history['val_loss'], label=key,color=color,zorder=zorder)

# \brief Forms the simple multilayer perceptron with provided parameters
#
# \param inputDimension - number of input nodes
# \param numClasses - number of output nodes
# \param lr - learning rate for the optimizer
# \param nodeNr- number of nodes in each layer
# \param layerNr- number of hidden layers
#
# \ret model
def formModel(inputDimension, numClasses, lr=0.001, nodeNr=20, layerNr=1, initialDropout=False, intermediateDropout=False, usedRegularizer=None):
    model = Sequential()
    
    # Input layer
    model.add(Input(inputDimension))
    if (initialDropout == True):
        model.add(Dropout(0.3))
        
    # Hidden layers
    for layer in range(layerNr):
        model.add(Dense(nodeNr, activation='relu')) # TODO: identify the best way to set kernel_regularizer=l2(1e-5)
        if (intermediateDropout == True):
            model.add(Dropout(0.3))
    
    # Output Layer
    model.add(Dense(numClasses, activation='softmax'))
    
    # Model parameters
    optimizer = Adam(lr)
    model.compile(loss='MSE', optimizer=optimizer, metrics=['accuracy']);
    return model

# \brief Forms the simple multilayer perceptron identical to one, used by Benjamin Timon.
#        All values are hardcoded. The input values are used only for compatibility purposes.
#
# \param inputDimension - number of input nodes. Ignored
# \param numClasses - number of output nodes. Ignored
# \param lr - learning rate for the optimizer. Ignored
# \param nodeNr- number of nodes in each layer. Ignored
# \param layerNr- number of hidden layers. Ignored
#
# \ret model
def formModelTimon(inputDimension=0, numClasses=0, lr=0, nodeNr=0, layerNr=0):
    model = Sequential()
    
    model.add(Input(inputDimension))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu')) 
    model.add(Dense(2, activation='softmax'))
    
    optimizer = Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-8, learning_rate=1e-3)
    model.compile(loss='MSE', optimizer=optimizer, metrics=['accuracy']);
    return model


# \brief Splits the array into 2, based on given condition
#
# \param arr - array to split
# \param cond - condition for splitting
#
# \ret model
def split(arr, cond):
    return [arr[cond], arr[~cond]]

# \brief Computes the signal-to-noise ratio for given dataset, splitted based on provided labels
#
# \param dataset - dataset for splitting
# \param labels - confition for splitting
#
# \ret model
def SNR(dataset,labels):
    
    # Split the sets according to computed labels
    set0, set1 = split(dataset, labels == 0) # TODO: construct better splitting function with multilabel support.
    
    # Compute signal(mean) for each set and use it to form the array of the same shape as original set
    signal0 = np.mean(set0, axis=0)
    signal1 = np.mean(set1, axis=0)
    signalShaped0 = np.repeat([signal0], repeats=set0[:, 0].size, axis=0)
    signalShaped1 = np.repeat([signal1], repeats=set1[:, 0].size, axis=0)
    
    # Compute noise
    noise0 = set0 - signalShaped0
    noise1 = set1 - signalShaped1
    
    # Combine the sets and compute variance of signal and noise.
    noise=np.concatenate((noise0,noise1))
    signal = np.concatenate((signalShaped0, signalShaped1))
    noiseVar=np.var(noise,axis=0) + 1e-100 # 1e-100 required to avoid division by zero. 
    signalVar = np.var(signal, axis=0)
    
    #compute SNR
    return signalVar/noiseVar

# \brief Performs the DPA(DDLA) attack on the provided data with respect to suggested configuration
#
# \param training_set data set, used to train the model. Map
# \param validation_set data set, used to compute the 
# \param config map, containing the values, required for iteration 
#
# \ret figure
def iterateModels(training_set,validation_set, config):

    for iteration in range (0,100): 
        for lr in range(3, 4):
            for epochs in range(100, 120, 20):
                for layerNr in range(1, 2):
                    for nodes in range(10, 20, 10):
                        config={
                            "learningRate" : lr,
                            "EpochsNumber" : epochs,
                            "LayersNumber" : layerNr,
                            "NodesNumber"  : nodes,
                            "BatchSize"    : config["BatchSize"],
                            "AttackedByte" : config["AttackedByte"],
                            "ClassNumber"  : 2, # For binary mapping (LSB/MSB)
                            "Step"         : config["Step"],
                            "Iteration"    : iteration,
                            "Traces"       : config["Traces"]
                            }
                    DPA(training_set["x"],training_set["y"],validation_set["x"],validation_set["y"],validation_set["key"],config)


# \brief Performs the DPA(DDLA) attack on the provided data with respect to suggested configuration
#
# \param traces - traces, used to train the model. Map
# \param plain - corresponding plaintexts, used to compute the labels
# \param traces_test - validation traces
# \param plain_test - validation plaintexts
# \param key_test - key, used to determine the performance of the model
# \param config - configuration dictionary, containing the additional paramteters
# \ret figure
def DPA(traces, plain, traces_test, plain_test, key_test, config):
    
    #Create labels for validation and convert to categorical
    labels_test = create_labels(plain_test, key_test[0,config["AttackedByte"]],keybytePos=config["AttackedByte"])
    cat_labels_test= to_categorical(labels_test,config["ClassNumber"])
    
    # prepare graph and model.
    fig,art=formGraph()
    model= formModelTimon(traces[0,:].size, numClasses=config["ClassNumber"], nodeNr=config["NodesNumber"], layerNr=config["LayersNumber"])
    model.save_weights(str(config["Iteration"])+'_iterated_model_weights.h5')
    
    
    for key in range (key_test[0,config["AttackedByte"]]-2,key_test[0,config["AttackedByte"]]+2):
          # For each key use the same initial weights to make the results more stable
          model.load_weights(str(config["Iteration"])+'_iterated_model_weights.h5')
          
          labels= create_labels(plain, key, keybytePos=config["AttackedByte"])
          
          cat_labels= to_categorical(labels,config["ClassNumber"])
          history = model.fit(x=traces, y=cat_labels, verbose = 1, epochs=config["EpochsNumber"], batch_size=config["BatchSize"], validation_data=(traces_test,cat_labels_test))
          plotStep(art,history,key,key_test[0,config["AttackedByte"]])
          print(key)

    fig.savefig("ASCAD_TIMON_MLP_RES/ASCAD_TIMON_MLP_20_10_"+str(config["Traces"])+"_"+str(config["Iteration"])+".png",dpi=1200)
    return fig


if __name__ == "__main__":
    
    # Required to reduce memory consumption
    plt.ioff()
    
    # Default parameters values
    config={
    "Path To Database" : "S:/ASCAD/ASCAD_data/ASCAD_data/ASCAD_databases/ASCAD.h5",
    "Attack Type"      : "DPA",
    "BatchSize"        : 1000,
    "Step"             : 100,
    "InitialWindowSize": 700,
    "AttackedByte"     : 2,
    "Traces"             : 1000,
    }

    training_set = {}
    profiling_set = {}
    validation_set = {}
    attack_set = {}
    #load traces
    (training_set["x"], training_set["y"], training_set["key"]), (attack_set["x"], attack_set["y"], attack_set["key"]) = load_ascad(config["Path To Database"], DPA_format = (config["Attack Type"]=="DPA"))

    # Extract the required number of traces
    profiling_set["x"] = training_set["x"][:config["Traces"],]
    profiling_set["y"] = training_set["y"][:config["Traces"],]
    profiling_set["key"] = training_set["key"][:config["Traces"],]
    
    # Execute the attack.
    with tf.device('/CPU:0'):
        fig= iterateModels(profiling_set,attack_set,config)
