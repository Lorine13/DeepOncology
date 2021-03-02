import os
from datetime import datetime
import math 
import tensorflow as tf

import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from experiments.exp_3d.preprocessing import * #getTransform
from lib.data_loader import DataGeneratorSurvival
from networks.VnetSurvival import *
from lib.DataManagerSurvival import DataManagerSurvival
from losses.LossMetricsSurvival import *

""" https://nbviewer.jupyter.org/github/sebp/survival-cnn-estimator/blob/master/tutorial_tf2.ipynb 
    https://github.com/chl8856/DeepHit
"""
#PARAMETRE MODEL SAVING
trained_model_path = None #If None, train from scratch 
training_model_folder = '../model/'
now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
training_model_folder = os.path.join(training_model_folder, now)  # '/path/to/folder'
if not os.path.exists(training_model_folder):
    os.makedirs(training_model_folder)
logdir = os.path.join(training_model_folder, 'logs')
if not os.path.exists(logdir):
    os.makedirs(logdir)

#PARAMETRES IMAGE PROCESSING
modalities = ('pet_img', 'ct_img') #input neural network ct and pet image
mask=False 
mode = ['binary', 'probs', 'mean_probs'][0]
method = ['otsu', 'absolute', 'relative', 'otsu_abs'][0]
#[if mode = binary & method = relative : t_val = 0.42
#if mode = binary & method = absolute : t_val = 2.5, 
#else : don't need tval]
tval = ''
target_size = (128, 128, 256)
target_spacing = (4.0, 4.0, 4.0)
target_direction = (1,0,0,0,1,0,0,0,1)

# PARAMETRES DATA MANAGER 
base_path='../../FLIP_NIFTI_COMPLET/'
excel_path='../FLIP.xlsx'
csv_path="../CSV_FLIP.csv"
create_csv = False

#PARAMETRE DATA GENERATOR
batch_size = 2
epochs = 100
shuffle = True 

#DATA MANAGER GET DATA
DM = DataManagerSurvival(base_path, excel_path,csv_path)
x,y = DM.get_data_survival(create_csv=create_csv)
x_train, x_val, y_train, y_val = DM.split_train_val_test_split(x, y, test_size=0.0, val_size=0.2, random_state=42)
train_images_paths_x, val_images_paths_x = DM.get_images_paths_train_val(x_train,x_val)

#IMAGE PROCESSING
train_transforms = get_transform('train', modalities, mask, mode, method, tval, target_size, target_spacing, target_direction, None, data_augmentation = True, from_pp=False, cache_pp=False)
val_transforms = get_transform('val', modalities, mask, mode, method, tval, target_size, target_spacing, target_direction, None,  data_augmentation = False, from_pp=False, cache_pp=False)

#DATA GENERATOR
train_generator = DataGeneratorSurvival(train_images_paths_x,
                                        y_train,
                                        train_transforms,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        x_key='input')

val_generator = DataGeneratorSurvival(val_images_paths_x,
                                        y_val,
                                        val_transforms,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        x_key='input')

#MULTI GPU TRAINING STRATEGY
strategy = tf.distribute.MirroredStrategy()

#################### MODEL LOSS METRICS OPTIMIZER ############################
time_horizon = math.ceil(max(y)[0]*1.2) #number of neurons on the output layer 

with strategy.scope():
    # definition of loss, optimizer and metrics
    loss_object = get_loss_survival(time_horizon_dim=time_horizon, batch_size=batch_size)
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    td_c_index = metric_td_c_index(time_horizon_dim=time_horizon, batch_size=batch_size)
    metrics = [] 

############################ MODEL CALLBACKS #################################
#PARAMETRES CALLBACKS
patience = 10
ReduceLROnPlateau1 = False
EarlyStopping1 = False
ModelCheckpoint1 = False
TensorBoard1 = True

callbacks = []
if ReduceLROnPlateau1 == True :
    # reduces learning rate if no improvement are seen
    learning_rate_reduction = ReduceLROnPlateau(monitor= loss_object,
                                                patience=patience ,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0000001)
    callbacks.append(learning_rate_reduction)

if EarlyStopping1 == True :
    # stop training if no improvements are seen
    early_stop = EarlyStopping(monitor="val_loss",
                                mode="min",
                                patience=int(patience // 2),
                                restore_best_weights=True)
    callbacks.append(early_stop)

if ModelCheckpoint1 == True :
    # saves model weights to file
    # 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(os.path.join(training_model_folder, 'model_weights.h5'),
                                    monitor=loss_object,  #td_c_index ?
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min',  # max
                                    save_weights_only=False)
    callbacks.append(checkpoint)

if TensorBoard1 == True :
    tensorboard_callback = TensorBoard(log_dir=logdir,
                                        histogram_freq=0,
                                        update_freq='epoch',
                                        write_graph=True,
                                        write_grads=True,
                                        write_images=True)
    callbacks.append(tensorboard_callback)

###################### DEFINITION MODEL ####################################
#PARAMETRES NEURAL NETWORK MODEL
architecture = 'vnet_survival'
image_shape= (256, 128, 128)
in_channels= len(modalities)
out_channels= 1
channels_last=True
keep_prob= 1.0
keep_prob_last_layer= 0.8
kernel_size= (5, 5, 5)
num_channels= 8
num_levels= 4
num_convolutions= (1, 2, 3, 3)
bottom_convolutions= 3
activation= "relu"
activation_last_layer= 'sigmoid'



with strategy.scope():
    model = VnetSurvival(image_shape,
            in_channels,
            out_channels,
            time_horizon,
            channels_last,
            keep_prob,
            keep_prob_last_layer,
            kernel_size,
            num_channels,
            num_levels,
            num_convolutions,
            bottom_convolutions,
            activation,
            activation_last_layer).create_model()


with strategy.scope():
    model.compile(loss=loss_object, optimizer=optimizer, metrics=metrics)

if trained_model_path is not None:
    with strategy.scope():
        model.load_weights(trained_model_path)


print(model.summary())

model_json = model.to_json()
with open(os.path.join(training_model_folder, 'architecture_{}_model_{}.json'.format(architecture, now)),
            "w") as json_file:
    json_file.write(model_json)

# training model
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    epochs=epochs,
                    callbacks=callbacks,  # initial_epoch=0,
                    verbose=1
                    )
                    
