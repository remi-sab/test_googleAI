# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:16:40 2020

@author: Broucki
"""

import os
import time
import pandas as pd
import numpy as np
import keras
import cv2
from math import ceil
import pydicom
from keras import backend as K
from keras.applications import ResNet50
from keras import layers
from pathlib import Path
import sklearn
import tensorflow as tf


start_time = time.time()

np.random.seed(2557)


destDirectory = '/test_google' #Folder contenant toutes les images

   

DATA_DIR = destDirectory
MODEL_NAME = 'model_one'
WEIGHTS_DIR = 'weights/' + MODEL_NAME + '/'
TB_DIR = 'tensorboard-graphs/Graph-' + MODEL_NAME
TB_FREQ = 67000

dir_path = Path(WEIGHTS_DIR)
dir_path2 = Path(TB_DIR)

#On supprimme le dossier model_one s'il existe
try:
    dir_path.rmdir()
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))
    
#On supprimme le dossier graph s'il existe
try:
    dir_path2.rmdir()
except OSError as e:
    print("Error: %s : %s" % (dir_path2, e.strerror))



os.mkdir(WEIGHTS_DIR)
os.mkdir(TB_DIR)

INPUT_SHAPE = (224, 224, 3)

def _read(path, desired_size):
    """Will be used in DataGenerator"""
    
    dcm = pydicom.dcmread(path)
    
    try:
        img = window_and_scale_brain_subdural_soft(dcm)
        img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)
        
    # Some dcms seem to be corrupted
    except ValueError:
        print('Error while parsing {}'.format(path))
        img = np.ones(desired_size)
    
    return img

class DataGenerator(keras.utils.Sequence):

    def __init__(self, img_dir, image_IDs, labels_df, batch_size, img_size):

        self.image_IDs = image_IDs
        self.labels_df = labels_df
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir

    def __len__(self):
        return int(ceil(len(self.image_IDs) / self.batch_size))

    def __getitem__(self, index):
        
        batch_ids = self.image_IDs[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.empty((self.batch_size, *self.img_size))
        Y = np.empty((self.batch_size, 6))
        
        for i, ID in enumerate(batch_ids):
            X[i,] = _read(self.img_dir+ID+".dcm", self.img_size)
            Y[i,] = self.labels_df.loc[ID].values
        
        return X, Y

# dcm processing

def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(dcm, window_center, window_width):
    
    #handle the 12 bit values
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img

def window_and_scale_brain_subdural_soft(dcm):
    
    #window images
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    bone_img = window_image(dcm, 600, 2800)
    
    #scale images (0-1)
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img + 20) / 200
    bone_img = (bone_img + 800) / 2800
    
    # combine channels
    return np.array([brain_img, subdural_img, bone_img]).transpose(1,2,0)



def read_trainset(filename=DATA_DIR+"../googleAI.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    # duplicates_to_remove = [
    #     56346,56347,56348,56349,
    #     56350,56351,1171830,1171831,
    #     1171832,1171833,1171834,1171835,
    #     3705312,3705313,3705314,3705315,
    #     3705316,3705317,3842478,3842479,
    #     3842480,3842481,3842482,3842483
    # ]
    
    # df = df.drop(index=duplicates_to_remove)
    # df = df.reset_index(drop=True)
    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    return df

# custom loss function
def weighted_log_loss(y_true, y_pred): #Binary cross entropy loss ponderee par le vecteur w (pour rÃ©duire class imbalance)
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """
    
    class_weights = np.array([1., 1., 1., 1., 1., 1.]) #Pas de poids
    
    eps = K.epsilon()
    
    y_pred = K.clip(y_pred, eps, 1.0-eps)

    out = -(         y_true  * K.log(      y_pred) * class_weights
            + (1.0 - y_true) * K.log(1.0 - y_pred) * class_weights)
    
    return K.mean(out, axis=-1)

def correct_positive_diagnoses(y_true, y_pred):
    THRESHOLD = 0.5
    p_thr = K.greater(y_pred, THRESHOLD)
    y_true = K.cast(y_true, dtype='bool')
    
    pos_mask = K.any(y_true, axis=1) #patients with positive diagnoses -> OR logique bitwise reduction
    p_thr = p_thr[pos_mask]
    y_true = y_true[pos_mask]
    
    equals_t = K.equal(p_thr, y_true)
    correct_rows = K.all(equals_t, axis=1) #-> AND logique bitwise reduction
    correct_rows_float = K.cast(correct_rows, dtype='float32')
    
    return K.sum(correct_rows_float)/(K.cast(K.shape(correct_rows_float)[0], dtype='float32')+K.epsilon())

from sklearn.model_selection import train_test_split

df = read_trainset()

train_df, test_df = train_test_split(df,test_size=0.2, random_state=257)

traingen = DataGenerator(img_dir=DATA_DIR,
                         image_IDs=train_df.index, #MAGIC
                         labels_df=train_df, #MAGIC
                         batch_size=16,
                         img_size=INPUT_SHAPE)

testgen = DataGenerator(img_dir=DATA_DIR,
                         image_IDs=test_df.index, #MAGIC
                         labels_df=test_df, #MAGIC
                         batch_size=16,
                         img_size=INPUT_SHAPE)

## Stats descriptives
#diagnoses = df['Label'].sum()/df['Label']['any'].sum()
#diagnoses = diagnoses.drop('any')

#%matplotlib inline
#import matplotlib.pyplot as plt
#plt.bar(diagnoses.keys(), height = diagnoses.values, color='orange')
#plt.xlabel('Diagnosis')
#plt.ylabel('Probability')
#plt.xticks(rotation='vertical')
#plt.show()

# Create a MirroredStrategy.
#strategy = tf.distribute.MirroredStrategy()
#print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
#with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
 #   model = get_compiled_model()
    


conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    
conv_base.trainable = True
model = keras.models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(6, activation='sigmoid'))
model._name = MODEL_NAME
model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(lr=1e-5),
    metrics=['accuracy'])

model.summary()
#mc = keras.callbacks.ModelCheckpoint(filepath=WEIGHTS_DIR+MODEL_NAME+'-epoch={epoch:02d}-valid-loss={val_loss:.2f}.hdf5', monitor='loss', verbose=True, save_best_only=False, save_weights_only=False)

#tb = keras.callbacks.TensorBoard(log_dir=TB_DIR, histogram_freq=0, update_freq=TB_FREQ, write_graph=True, write_images=True) # about 10 checkpoints per epoch
### ESSAI AVEC BINARY CROSSENTROPY ET accuracy ###
hist = model.fit_generator(traingen,
                    validation_data = testgen,
                    epochs=5,
                    verbose=True)
                    #workers=4,
                    #callbacks=[mc, tb])
                    #use_multiprocessing=True,


### Analyse du rappel, precision et justesse en faisant varier la valeur du seuil ###
p = model.predict_generator(testgen)
#p #2 mini batches

truth = np.array(testgen.labels_df, dtype='bool')
#truth #Les vrais labels de test

pred = p[:len(truth)] #Labels prÃ©dit de test, on garde le nombre de prediction reel
#pred #Labels predit de test, on prend tout le batch

label_names = df['Label'].keys().to_numpy() #On garde en memoire les labels
#label_names

thr = 0

while thr < 1.01 :
    pred_bool = pred > thr #(en prenant 0.5, on obtient False partout)
 
    print("Rappel pour seuil : "+str(round(thr,2)))
    recalls = []
    for i, lab in enumerate(label_names):
        recall = sklearn.metrics.recall_score(truth[:,i], pred_bool[:,i])
        recalls.append(recall)
        print(f'Recall for {lab}: {recall}')
    print('Rappel moyen: '+str(np.mean(recalls)))    
    print('')
    
    print("Precision pour seuil : "+str(round(thr,2)))    
    precisions = []
    for i, lab in enumerate(label_names):
        precision = sklearn.metrics.precision_score(truth[:,i], pred_bool[:,i])
        precisions.append(precision)
        print(f'Precision for {lab}: {precision}')
    print('Precision moyenne: '+str(np.mean(precisions)))
    print('')
            
    thr = thr+0.05

best_thr = 0.2 #Remplacer par la valeur seuille souhaitee
pred_best = pred > best_thr 

recalls = []
for i, lab in enumerate(label_names):
    recall = sklearn.metrics.recall_score(truth[:,i], pred_best[:,i])
    recalls.append(recall)
    print(f'Recall for {lab}: {recall}')
    
#################################################################################################

### Etude sur les any ###

any_truth = testgen.labels_df['Label']['any'].values #Vraies valeurs de la variable any
#any_truth

any_explicit = np.array(np.sum(pred_best[:,1:], axis=1), dtype=np.bool) #On regarde si il y a un 1 dans les categories
any_predicted = pred_best[:,0] #On regarde juste la colonne 0

#any_explicit

#any_predicted

# accuracy comparison
print('')
print("Acc predicted:")
print(sklearn.metrics.accuracy_score(any_truth, any_predicted))
print("Acc explicit:")
print(sklearn.metrics.accuracy_score(any_truth, any_explicit))

# recall comparison
print('')
print("Rappel predicted:")
print(sklearn.metrics.recall_score(any_truth, any_predicted))
print("Rappel explicit:")
print(sklearn.metrics.recall_score(any_truth, any_explicit))

# precision comparison
print('')
print("Precision predicted:")
print(sklearn.metrics.precision_score(any_truth, any_predicted))
print("Precision explicit:")
print(sklearn.metrics.precision_score(any_truth, any_explicit))

################################################################################################

final_time = time.time()
print('')
print("Temps: "+str(final_time-start_time))