import tensorflow as tf
import numpy as np
import math 
from lifelines import KaplanMeierFitter

e=[[7,3,10,6],[1,1,1,0]]
x=tf.constant(e)
p=tf.multiply(x[0],x[1])

num_Category=13
time=x.numpy()[0]
event=x.numpy()[1]

mask = np.zeros([len(time), num_Category]) # for the first loss function
for i in range(len(time)):
    if event[i] != 0:  #not censored
        mask[i,int(time[i])] = 1
    else: #censored 
        mask[i,int(time[i]+1):] = 1 #fill 1 until from the censoring time (to get 1 - \sum F)
#print(mask) 
y_pred= tf.constant([[0.1,0.0,0.1,0.1,0.1,0.1,0.1,0.3,0.0,0.0,0.0,0.1,0.0],
                    [0.1,0.0,0.4,0.0,0.1,0.1,0.0,0.1,0.0,0.0,0.0,0.1,0.1],
                    [0.0,0.0,0.0,0.1,0.1,0.1,0.1,0.0,0.0,0.2,0.3,0.1,0.0],
                    [0.1,0.0,0.0,0.1,0.1,0.1,0.3,0.1,0.0,0.0,0.0,0.1,0.1]])
tmp1 = event*tf.math.log(tf.reduce_sum(mask * y_pred, 1))
tmp2=(1-event)*tf.math.log(tf.reduce_sum( mask* y_pred,1))
loss=-tf.reduce_sum(tmp1+tmp2)
#print(loss)
#tmp2 = (1. - I_1) * log(tmp2)
mask = np.zeros([len(time), num_Category])
for i in range(len(time)):
    t = int(time[i]) # censoring/event time
    mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    

print(tf.transpose(mask))

#################LOSS 2#########################   
# PARAMETRES 
num_Category=13
time=x.numpy()[0]
event=x.numpy()[1]
y_pred=tf.cast(y_pred,tf.float32)

sigma = tf.constant(0.1, dtype=tf.float32)
one_vector = tf.ones_like([time], dtype=tf.float32)
event_tf = tf.cast(event, dtype = tf.float32) #indicator for event
time_tf = tf.cast([time],dtype=tf.float32)

mask=tf.cast(tf.transpose(mask),tf.float32)
I_2 = tf.linalg.diag(event_tf)

R = tf.linalg.matmul(y_pred, mask)
diag_R = tf.linalg.diag_part(R)
diag_R = tf.reshape(diag_R, [-1, 1])
R2 = tf.transpose(diag_R-R)

T = tf.nn.relu(tf.sign(tf.matmul(tf.transpose(one_vector), time_tf)-tf.matmul(tf.transpose(time_tf), one_vector)))
T2 = tf.matmul(I_2, T)
loss = tf.reduce_sum(tf.reduce_mean(T2 * tf.exp(-R2/sigma),1, keepdims=True))
#print("ouiii")
#R2 = tf.linalg.matmul(one_vector2, diag_R)
#R3= R2 - R
#tmp_e = tf.reshape(tf.slice(y_pred, [0, 0], [-1, 1]), [-1, num_Category]) #event specific joint prob.
#n=- tf.reduce_sum(tmp1)

##################################################
kmf = KaplanMeierFitter()
kmf.fit(time, event.astype(int))  # censoring prob = survival probability of event "censoring"
G = np.asarray(kmf.survival_function_.reset_index()).transpose()
G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)

##################################################
num_Category=13
time=x.numpy()[0]
event=x.numpy()[1]
y_pred=tf.cast(y_pred,tf.float32)
eval_time= 6
'''
    This is a cause-specific c(t)-index
    - Prediction      : risk at Time (higher --> more risky)
    - Time_survival   : survival/censoring time
    - Death           :
        > 1: death
        > 0: censored (including death from other cause)
    - Time            : time of evaluation (time-horizon when evaluating C-index)
'''
y_pred=y_pred.numpy()
prediction_t= np.zeros(len(y_pred))
prediction = np.sum(y_pred[:,:(eval_time+1)], axis=1)
print(y_pred)
print(prediction)
N = len(y_pred)
A = np.zeros((N,N))
Q = np.zeros((N,N))
N_t = np.zeros((N,N))
Num = 0
Den = 0
for i in range(N):
    tmp_idx = np.where(G[0,:] >= time[i])[0]
    print("nooon")
    print(tmp_idx)
    A[i, np.where(time[i] < time)] = 1
    Q[i, np.where(prediction[i] > prediction)] = 1

    if (time[i]<=eval_time and event[i]==1):
        N_t[i,:] = 1
print(A)
print(Q)
print(N_t)
Num  = np.sum(((A)*N_t)*Q)
Den  = np.sum((A)*N_t)

########################################


print("oui")