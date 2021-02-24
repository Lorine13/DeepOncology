import glob 
import numpy as np
import os
from datetime import date
import openpyxl
import math 
import tensorflow_addons as tfa
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split

from experiments.exp_3d.preprocessing import *
from dataGenerator import *
from .Vnet_survival import *

""" https://nbviewer.jupyter.org/github/sebp/survival-cnn-estimator/blob/master/tutorial_tf2.ipynb 
    https://github.com/chl8856/DeepHit
"""

#PARAMETRES 
csv_path = ''
pp_dir = ''
modalities = ('pet_img', 'ct_img')
mode = ['binary', 'probs', 'mean_probs'][0]
method = ['otsu', 'absolute', 'relative', 'otsu_abs'][0]

#callbacks
patience = 10
ReduceLROnPlateau = False
EarlyStopping = False
ModelCheckpoint = True
TensorBoard = True

#parameters
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

architecture = 'vnet_survival'
#[if mode = binary & method = relative : t_val = 0.42
#if mode = binary & method = absolute : t_val = 2.5, 
#else : don't need tval]
tval = ''
target_size = (128, 128, 256)
target_spacing = (4.0, 4.0, 4.0)
target_direction = (1,0,0,0,1,0,0,0,1)

from_pp=False
cache_pp=False
pp_flag=''

mask=False

###Parametres dataGenerator 
epochs = 100
batch_size = 2
shuffle = True 

#FIN PARAMETRES 

def get_data_survival(base_path, csv_path, excel_path):
    y_time=[]
    y_event=[]
    PT_paths=[]
    CT_paths=[]
    data_exists= True
    excel = openpyxl.load_workbook('FLIP.xlsx')
    sheet = excel.active

    for i in range(2, sheet.max_row):
        data_exists=True
        x = sheet.cell(row=i, column=5)
        if (x.value != None):
            #retrieve the path to the nifti files from the patient : folder with the Anonymisation name from the excel 
            path=base_path+sheet.cell(row=i, column=2).value+'/'
            nifti_path= glob.glob(os.path.join(path, '**/*_nifti_PT.nii'), recursive=True)
            if (nifti_path):
                PT_paths=np.append(PT_paths, [nifti_path[0]])
                nifti_path2= glob.glob(os.path.join(path, '**/*_nifti_CT.nii'), recursive=True)
                if ( nifti_path2):
                    CT_paths=np.append(CT_paths, [nifti_path2[0]])
                else:
                    data_exists=False
            else:
                data_exists=False

            if data_exists:
                #retrieve y_time and y_event from excel 

                x = x.value.split('/')
                x = [int(i) for i in x]
                diagnosis_date= date(x[2],x[0],x[1])
                if (sheet.cell(row=i, column=7).value==0 and sheet.cell(row=i, column=9).value!=None):
                    y= sheet.cell(row=i, column=9).value.split('/')
                    y = [int(j) for j in y]
                    last_checkup_date= date(y[2],y[0],y[1])
                elif (sheet.cell(row=i, column=7).value==1 and sheet.cell(row=i, column=8).value!=None):
                    y= sheet.cell(row=i, column=8).value.split('/')
                    y = [int(j) for j in y]
                    last_checkup_date= date(y[2],y[0],y[1])
                
                time= int(((last_checkup_date-diagnosis_date).days)/30)
                y_time=np.append(y_time, [time])
                y_event=np.append(y_event,[int(sheet.cell(row=i, column=7).value)])



    y_event= y_event.astype(np.int32)
    y_time=y_time.astype(np.int32)
    return list(zip(PT_paths, CT_paths)), list(zip(y_time, y_event))

base_path='../../FLIP_NIFTI/'
x,y= get_data_survival(base_path,"","")

#number of neurons on the output layer 
time_horizon = math.ceil(max(y)[0]*1.2)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

train_y_time=[y_t[0] for y_t in y_train]
val_y_time=[y_v[0] for y_v in y_val]
train_y_event=[y_t[1] for y_t in y_train]
val_y_event=[y_v[1] for y_v in y_val]

y_train=[train_y_time, train_y_event]
y_val=[val_y_time, val_y_event]

dataset_x = dict()
dataset_x['train']=[]
dataset_x['val']=[]

for i in range(len(x_train)):
    dataset_x['train'].append({'pet_img':x_train[i][0], 'ct_img':x_train[i][1]})
for i in range(len(x_val)):    
    dataset_x['val'].append({'pet_img':x_val[i][0], 'ct_img':x_val[i][1]})

train_transforms = get_transform('train', modalities, mask, mode, method, tval, target_size, target_spacing, target_direction, None, data_augmentation = True, from_pp=False, cache_pp=False)
val_transforms = get_transform('val', modalities, mode, mask, method, tval, target_size, target_spacing, target_direction, None,  data_augmentation = False, from_pp=False, cache_pp=False)

# multi gpu training strategy
strategy = tf.distribute.MirroredStrategy()

train_images_paths_x, val_images_paths_x = dataset_x['train'], dataset_x['val']

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

#################################################################################

with strategy.scope():
    # definition of loss, optimizer and metrics
    loss_object = get_loss_survival(time_horizon_dim=time_horizon)
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    td_c_index = td_c_index(dim=3, vnet=True)
    metrics = [td_c_index] 

# callbacks
callbacks = []
if ReduceLROnPlateau == True :
    # reduces learning rate if no improvement are seen
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                patience=patience ,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0000001)
    callbacks.append(learning_rate_reduction)

if EarlyStopping == True :
    # stop training if no improvements are seen
    early_stop = EarlyStopping(monitor="val_loss",
                                mode="min",
                                patience=int(patience // 2),
                                restore_best_weights=True)
    callbacks.append(early_stop)

if ModelCheckpoint == True :
    # saves model weights to file
    # 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(os.path.join(training_model_folder, 'model_weights.h5'),
                                    monitor='val_loss',  
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min',  # max
                                    save_weights_only=False)
    callbacks.append(checkpoint)

if TensorBoard == True :
    tensorboard_callback = TensorBoard(log_dir=logdir,
                                        histogram_freq=0,
                                        batch_size=batch_size,
                                        write_graph=True,
                                        write_grads=True,
                                        write_images=False)
    callbacks.append(tensorboard_callback)

# Define model
if architecture.lower() == 'vnet':
    with strategy.scope():
        model = VNetSurvival(image_shape,
                in_channels,
                out_channels,
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
else:
    raise ValueError('Architecture ' + architecture + ' not supported. Please ' +
                        'choose one of unet|vnet.')
with strategy.scope():
    model.compile(loss=loss_object, optimizer=optimizer, metrics=metrics)

#if trained_model_path is not None:
#    with strategy.scope():
#        model.load_weights(trained_model_path)

print(model.summary())

"""
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        epochs=epochs,
                        callbacks=callbacks,  # initial_epoch=0,
                        verbose=1
                        )
"""
