import tensorflow as tf
import numpy as np
import math 
from lifelines import KaplanMeierFitter

e=[[7,3,10,6],[0,0,1,0]]
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
""""
time_horizon_dim=13
print("oui")
########################################
mask = tf.zeros([time.shape, time_horizon_dim]) 
for i in range(len(time)):
    if event[i] != 0:  #not censored
        mask[i,int(time[i])] = 1
    else: #censored 
        mask[i,int(time[i]+1):] = 1 #fill 1 until from the censoring time (to get 1 - \sum F)

#for uncensored: 
tmp1 = event*tf.math.log(tf.reduce_sum(mask * y_pred, 1))
#for censored: log \sum P(T>t|x)
tmp2=(1-event)*tf.math.log(tf.reduce_sum( mask* y_pred,1))

loss=-tf.reduce_sum(tmp1+tmp2)
"""
############################################
""""def get_logLikelihood_LOSS(time, event, time_horizon_dim, y_pred,batch_size):


    #mask for the log-likelihood loss
    #mask size is [N, time_horizon_dim]
    #    if not censored : one element = 1 (0 elsewhere)
    #    if censored     : fill elements with 1 after the censoring time (for all events)
    #mask = np.zeros([len(time), time_horizon_dim]) 
    print("ouiiiiiiiiiiiii")
    print(len(time))
    mask = np.zeros([batch_size, time_horizon_dim]) 
    print(tf.one_hot(time, time_horizon_dim))
    for i in range(batch_size):
        if event[i] != 0:  #not censored
            mask[i,int(time[i])] = 1
        else: #censored 
            mask[i,int(time[i]+1):] = 1 #fill 1 until from the censoring time (to get 1 - \sum F)
    
    #for uncensored: 
    tmp1 = event*tf.math.log(tf.reduce_sum(mask * y_pred, 1))
    #for censored: log \sum P(T>t|x)
    tmp2=(1-event)*tf.math.log(tf.reduce_sum( mask* y_pred,1))
    
    loss=-tf.reduce_sum(tmp1+tmp2)

    return loss
    
y_pred= tf.constant([[0.1,0.0,0.1,0.1,0.1,0.1,0.1,0.3,0.0,0.0,0.0,0.1,0.0],
                [0.1,0.0,0.4,0.0,0.1,0.1,0.0,0.1,0.0,0.0,0.0,0.1,0.1],
                [0.0,0.0,0.0,0.1,0.1,0.1,0.1,0.0,0.0,0.2,0.3,0.1,0.0],
                [0.1,0.0,0.0,0.1,0.1,0.1,0.3,0.1,0.0,0.0,0.0,0.1,0.1]])
a = tf.Variable(tf.zeros([len(time), num_Category]) )
x=tf.constant(e)[0]
#a=tf.cast(tf.assign(a[:,x[0]:], tf.ones_like(a[:,x[i]:])))
#a[:,x[0]:].assign(1)
print(tf.zeros([1,time[0]]))
a=tf.zeros([1,time[0]],dtype=tf.int32)
b=tf.ones([1,13-time[0]],dtype=tf.int32)
c= tf.concat([a,b],1)
for i in range(len(time)):
    a=tf.ones([1,time[0]],dtype=tf.int32)
    b=tf.zeros([1,13-time[0]],dtype=tf.int32)
    d= tf.concat([a,b],1)
d=tf.cast(d, dtype=tf.float32)
oui=y_pred*d
print(d)
print(y_pred)
print(oui)
non=tf.math.reduce_sum(oui,1)
print(non)
nb = 0
#print(c)
#print(b)
results=tf.constant([])
time_horizon_dim=13
for eval_time in range (time_horizon_dim):
    a=tf.zeros([1,eval_time],dtype=tf.int32)
    b=tf.ones([1,time_horizon_dim-eval_time],dtype=tf.int32)
    c= tf.concat([a,b],1)
    for i in range(eval_time):
        a=tf.ones([1,eval_time],dtype=tf.int32)
        b=tf.zeros([1,13-eval_time],dtype=tf.int32)
        c= tf.concat([a,b],1)
    c=tf.cast(c, dtype=tf.float32)
    d=y_pred*c
    pred_t=tf.math.reduce_sum(d,1, keepdims=True)
    for i in range(len(time)):
        A=tf.math.greater(time, time[i])
        print("ouiiiiiiiiiiiiiiiiii")
        print(A)
        A=tf.where(A,1,0)
        Q=tf.math.greater(pred_t[i],pred_t)
        Q=tf.where(Q,1,0)
        
        if (time[i]<=eval_time and event[i]==1):
            N_t=tf.ones([1,len(time)])
        else:
            N_t=tf.zeros([1,len(time)])
            #N_t[i,:] = 1
            
        if i==0:
            mat_A=A
            mat_Q=Q
            mat_N_t=N_t
        else:
            mat_A=tf.concat([mat_A,A],0)
            mat_Q=tf.concat([mat_Q,Q],0)
            mat_N_t=tf.concat([mat_N_t,N_t],0)
    mat_A=tf.reshape(mat_A, [len(time),len(time)])
    mat_Q=tf.reshape(mat_Q, [len(time),len(time)])
    mat_N_t=tf.reshape(mat_N_t, [len(time),len(time)])

    mat_A=tf.cast(mat_A,dtype=tf.float32)
    mat_Q=tf.cast(mat_Q,dtype=tf.float32)
    mat_N_t=tf.cast(mat_N_t,dtype=tf.float32)
    Num= tf.reduce_sum((mat_A*mat_N_t)*mat_Q)
    Den=tf.reduce_sum(mat_A*mat_N_t)
    #Num  = np.sum(((A)*N_t)*Q)
    #Den  = np.sum((A)*N_t)
    print("num")
    print(Num)
    print("den")
    print(Den)
    if Num == 0 and Den == 0:
        #results = -1 # not able to compute c-index!
        print("je suis là")
    else:
        if tf.equal(tf.size(results),0):
            print("et la je suis par làààà22222222222222")
            results = tf.constant([float(Num/Den)])
            nb+=1
        else:
            nb+=1
            print("et la je suis par làààà")
            print(results)
            truc = tf.constant([float(Num/Den)])
            print(truc)
            #results.append(float(Num/Den))
            results=tf.math.add(truc, results)
            #results= tf.concat([results, truc],0)
            #results= tf.reduce_sum(results)
            print(results)
    #print(results)
result = results/nb
print("ouiiiii")
print(result)

"""
#####################################################"""""""""
batch_size=4
time=tf.constant(e)[0]
event=tf.constant(e)[1]
time_horizon_dim=13
for i in range(batch_size):
    if event[i] != 0:  #not censored
        if i==0:
            print("mask1")
            print(mask)
            mask=tf.one_hot(time[i], time_horizon_dim,dtype=tf.int32)
            print("mask1")
            print(mask)
        else:
            print("mask2")
            print(mask)
            print(tf.one_hot(time[i], time_horizon_dim,dtype=tf.int32))
            mask=tf.concat([tf.squeeze(mask), tf.one_hot(time[i], time_horizon_dim,dtype=tf.int32)],0)
        #mask[i,int(time[i])] = 1
            print("mask2")
            print(mask)
    else: #censored 
        if i==0:
            print("mask3")
            print(mask)
            a=tf.zeros([1,time[0]],dtype=tf.int32)
            b=tf.ones([1,time_horizon_dim-time[0]],dtype=tf.int32)
            mask=tf.concat([a,b],1)
            print("mask3")
            print(mask)
        elif i!=0:
            print("mask4")
            a=tf.zeros([1,time[0]],dtype=tf.int32)
            b=tf.ones([1,time_horizon_dim-time[0]],dtype=tf.int32)
            c=tf.concat([a,b],1)
            c=tf.squeeze(c)
            print(a)
            print(b)
            print(c)
            mask=tf.concat([tf.squeeze(mask), c],0)
            print("mask4")
            print(mask)

        #mask[i,int(time[i]+1):] = 1 #fill 1 until from the censoring time (to get 1 - \sum F)
print("this is the mask ")
print(mask)
mask = tf.reshape(mask,[batch_size, time_horizon_dim])
print(mask)
#for uncensored: 
event=tf.cast(event, dtype=tf.float32)
mask=tf.cast(mask, dtype=tf.float32)
oui= tf.reduce_sum(mask * y_pred, 1)
print("oui")
print(oui)
oui=tf.where(tf.equal(oui,0), 0.0001, oui)
print(oui)
tmp1 = event*tf.math.log(oui)
#for censored: log \sum P(T>t|x)
tmp2=(1-event)*tf.math.log(oui)
print(tmp1)
print(tmp2)

loss=-tf.reduce_sum(tmp1+tmp2)
print(loss)
print("oui")
