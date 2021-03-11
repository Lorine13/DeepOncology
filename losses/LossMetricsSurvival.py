import tensorflow as tf
import numpy as np
import sys
from lifelines import KaplanMeierFitter
#import sksurv.metrics.concordance_index_ipcw

def get_brier_loss(time_horizon_dim, batch_size):

    def brier_score(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        time=y_true[0]
        event=y_true[1]
        event=tf.cast(event, dtype=tf.float32)
        ones= tf.one_hot(time, time_horizon_dim)
        preds= tf.reduce_sum(y_pred*ones, 1, keepdims=True)
        preds= (preds-1)                           
        
        preds=preds**2
        preds= preds*tf.reshape(event,[batch_size,1])
        return tf.reduce_mean(preds)
    return brier_score

def get_logLikelihood_LOSS(time, event, time_horizon_dim, y_pred,batch_size):
    '''
    for uncensored : log of the probabilitie predicted to experience the event at the true time t
    for censored : log of the sum of probabilities predicted to experience the event after the true censoring time t
    '''

    #mask for the log-likelihood loss
    #mask size is [N, time_horizon_dim]
    #    if not censored : one element = 1 (0 elsewhere)
    #    if censored     : fill elements with 1 after the censoring time (for all events)
    #mask = np.zeros([len(time), time_horizon_dim]) 
    for i in range(batch_size):
        if event[i] != 0:  #not censored
            if i==0:
                mask=tf.one_hot(time[i], time_horizon_dim,dtype=tf.int32)
            else:
                mask=tf.concat([tf.squeeze(mask), tf.one_hot(time[i], time_horizon_dim,dtype=tf.int32)],0)
            #mask[i,int(time[i])] = 1
        else: #censored 
            if i==0:
                a=tf.zeros([1,time[i]],dtype=tf.int32)
                b=tf.ones([1,time_horizon_dim-time[i]],dtype=tf.int32)
                mask=tf.concat([a,b],1)
            elif i!=0:
                a=tf.zeros([1,time[i]],dtype=tf.int32)
                b=tf.ones([1,time_horizon_dim-time[i]],dtype=tf.int32)
                c=tf.concat([a,b],1)
                c=tf.squeeze(c)
                mask=tf.concat([tf.squeeze(mask), c],0)

            #mask[i,int(time[i]+1):] = 1 #fill 1 until from the censoring time (to get 1 - \sum F)
    
    #for uncensored: 

    mask = tf.reshape(mask,[batch_size, time_horizon_dim])
    event=tf.cast(event, dtype=tf.float32)
    mask=tf.cast(mask, dtype=tf.float32)

    oui= tf.reduce_sum(mask * y_pred, 1)
    oui=tf.where(tf.equal(oui,0), 0.0001, oui)
    tmp1 = event*tf.math.log(oui)
    #for censored: log \sum P(T>t|x)
    tmp2=(1-event)*tf.math.log(oui)
    
    #loss=-tf.reduce_sum(tmp1+tmp2)
    loss=-tf.reduce_mean(tmp1+tmp2)
    return loss

def get_ranking_LOSS(time, event, time_horizon_dim, y_pred, batch_size):
    '''
    for pairs acceptables (careful with censored events):
    loss  function η(P(x),P(y)) = exp(−(P(x)−P(y))/σ)
    where P(x) is the sum of probabilities for x to experience the event on time t <= tx -- (true time of x) --
    and P(y) is the sum of probabilities for y to experience the event on time t <= tx
    translated to : a patient who dies at a time s should have a higher risk at time s than a patient who survived longer than s
    '''
    #    mask is required calculate the ranking loss (for pair-wise comparision)
    #    mask size is [N, time_horizon_dim].
    #         1's from start to the event time(inclusive)
    #mask = np.zeros([len(time), time_horizon_dim])
    for i in range(batch_size):
        if i==0:
            a=tf.ones([1,time[i]+1],dtype=tf.int32)
            b=tf.zeros([1,time_horizon_dim-(time[i]+1)],dtype=tf.int32)
            mask=tf.concat([a,b],1)
        else: 
            a=tf.ones([1,time[i]+1],dtype=tf.int32)
            b=tf.zeros([1,time_horizon_dim-(time[i]+1)],dtype=tf.int32)
            c=tf.concat([a,b],1)
            mask=tf.concat([mask,c],0)
        #t = int(time[i]) # censoring/event time
        #mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    
    sigma = tf.constant(0.1, dtype=tf.float32)
    one_vector = tf.ones_like([time], dtype=tf.float32)
    event_tf = tf.cast(event, dtype = tf.float32) #indicator for event
    time_tf = tf.cast([time],dtype=tf.float32)

    mask=tf.cast(tf.transpose(mask),tf.float32)
    I_2 = tf.linalg.diag(event_tf)
    
    #R : r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})
    R = tf.linalg.matmul(y_pred, mask)
    diag_R = tf.linalg.diag_part(R)
    diag_R = tf.reshape(diag_R, [-1, 1]) # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
    R2 = tf.transpose(diag_R-R)
    # diag_R-R : R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
    # transpose : R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

    T = tf.nn.relu(tf.sign(tf.matmul(tf.transpose(one_vector), time_tf)-tf.matmul(tf.transpose(time_tf), one_vector)))
    # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

    T2 = tf.matmul(I_2, T)
    # only remains T_{ij}=1 when event occured for subject i
    loss = tf.reduce_mean(T2 * tf.exp(-R2/sigma)                        
    
    
    )
    #loss = tf.reduce_sum(tf.reduce_mean(T2 * tf.exp(-R2/sigma),1, keepdims=True))
    #loss = tf.reduce_sum(tf.reduce_mean(1+(T2*tf.log(-sigma*R2)/log(2))) ###from the rnn surv paper supposedly less expensive computionnaly 

    return loss

def get_loss_survival(time_horizon_dim, batch_size, alpha, beta, gamma):
    ''' 
    time_horizon_dim : output dimension of the output layer of the model
    loss_survival : returns the loss of the model (log_likelihood loss + ranking loss)
    '''
    def loss_survival(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        time=y_true[0]
        event=y_true[1]

        loss_logLikelihood= get_logLikelihood_LOSS(time, event, time_horizon_dim, y_pred, batch_size)
        loss_ranking=get_ranking_LOSS(time, event, time_horizon_dim, y_pred, batch_size)
        #loss_brier=get_brier_loss(time,event, time_horizon_dim, y_pred,batch_size)
        
        loss= alpha*loss_logLikelihood + beta*loss_ranking #+gamma*loss_brier
        return loss

    return loss_survival

def metric_td_c_index(time_horizon_dim,batch_size):
    
    def td_c_index(y_true, y_pred):
        time=y_true[0]
        event=y_true[1]
        '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
        '''
        nb=0
        resultat=0.0
        for eval_time in range (time_horizon_dim):
            #results=tf.constant([])
            #a=tf.zeros([1,eval_time],dtype=tf.int32)
            #b=tf.ones([1,time_horizon_dim-eval_time],dtype=tf.int32)
            #c= tf.concat([a,b],1)
            a=tf.ones([1,eval_time],dtype=tf.int32)
            b=tf.zeros([1,time_horizon_dim-eval_time],dtype=tf.int32)
            c= tf.concat([a,b],1)
            c=tf.cast(c, dtype=tf.float32)
            d=y_pred*c
            pred_t=tf.math.reduce_sum(d,1, keepdims=True)
            
            for i in range(batch_size):
                A=tf.math.greater(time, time[i])
                A=tf.where(A,1,0)
                Q=tf.math.greater(pred_t[i],pred_t)
                Q=tf.where(Q,1,0)
                #A[i, np.where(time[i] < time)] = 1
                #Q[i, np.where(pred_t[i] > pred_t)] = 1

                if (time[i]<=eval_time and event[i]==1):
                    N_t=tf.ones([1,batch_size])
                else:
                    N_t=tf.zeros([1,batch_size])
                    #N_t[i,:] = 1
                    
                if i==0:
                    mat_A=A
                    mat_Q=Q
                    mat_N_t=N_t
                else:
                    mat_A=tf.concat([mat_A,A],0)
                    mat_Q=tf.concat([mat_Q,Q],0)
                    mat_N_t=tf.concat([mat_N_t,N_t],0)
            
            mat_A=tf.reshape(mat_A, [batch_size,batch_size])
            mat_Q=tf.reshape(mat_Q, [batch_size,batch_size])
            mat_N_t=tf.reshape(mat_N_t, [batch_size,batch_size])

            mat_A=tf.cast(mat_A,dtype=tf.float32)
            mat_Q=tf.cast(mat_Q,dtype=tf.float32)
            mat_N_t=tf.cast(mat_N_t,dtype=tf.float32)

            Num= tf.reduce_sum((mat_A*mat_N_t)*mat_Q)
            Den=tf.reduce_sum(mat_A*mat_N_t)
            #tf.print(mat_A, output_stream=sys.stdout)
            #Num  = np.sum(((A)*N_t)*Q)
            #Den  = np.sum((A)*N_t)
            if Num != 0.0 and Den != 0.0:
                nb+=1
                resultat+=float(Num/Den)
        if resultat!=0:
            resultat = resultat/float(nb)
        return float(resultat)
    return td_c_index

def metric_cindex(time_horizon_dim,batch_size, tied_tol=1e-2):
    
    def cindex(y_true, y_pred):
        time=y_true[0]
        event=y_true[1]
        '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
        '''
        nb=0
        resultat=0.0
        num_tied_tot=0
        num_tied_true=0
        for i in range (batch_size):
            #results=tf.constant([])
            #a=tf.zeros([1,eval_time],dtype=tf.int32)
            #b=tf.ones([1,time_horizon_dim-eval_time],dtype=tf.int32)
            #c= tf.concat([a,b],1)
            a=tf.ones([1,time[i]],dtype=tf.int32)
            b=tf.zeros([1,time_horizon_dim-time[i]],dtype=tf.int32)
            c= tf.concat([a,b],1)
            c=tf.cast(c, dtype=tf.float32)
            d=y_pred*c
            pred_t=tf.math.reduce_sum(d,1, keepdims=True)
            
            ##########################
            time_equal= tf.math.equal(time[i], time)
            time_equal= tf.where(time_equal,1,0)
            time_equal=tf.cast(time_equal, dtype=tf.int32)
            event=tf.cast(event, dtype=tf.int32)
            time_equal= time_equal*event
            num_tied_tot+=tf.reduce_sum(time_equal[i+1:])
            pred_equal = tf.math.less(abs(pred_t[i]-pred_t), tied_tol)
            pred_equal = tf.where(pred_equal, 1,0)
            ties = tf.transpose(pred_equal)*time_equal
            num_tied_true+=tf.reduce_sum(ties[:,i+1:])
            ##########################

            A=tf.math.greater(time, time[i])
            A=tf.where(A,1,0)
            Q=tf.math.greater(pred_t[i],pred_t)
            Q=tf.where(Q,1,0)

            if (event[i]==1):
                N_t=tf.ones([1,batch_size])
            else:
                N_t=tf.zeros([1,batch_size])
                
            if i==0:
                mat_A=A
                mat_Q=Q
                mat_N_t=N_t
            else:
                mat_A=tf.concat([mat_A,A],0)
                mat_Q=tf.concat([mat_Q,Q],0)
                mat_N_t=tf.concat([mat_N_t,N_t],0)
        
        mat_A=tf.reshape(mat_A, [batch_size,batch_size])
        mat_Q=tf.reshape(mat_Q, [batch_size,batch_size])
        mat_N_t=tf.reshape(mat_N_t, [batch_size,batch_size])

        mat_A=tf.cast(mat_A,dtype=tf.float64)
        mat_Q=tf.cast(mat_Q,dtype=tf.float64)
        mat_N_t=tf.cast(mat_N_t,dtype=tf.float64)

        Num= tf.reduce_sum((mat_A*mat_N_t)*mat_Q)+(int(num_tied_true)/2)
        Den=tf.reduce_sum(mat_A*mat_N_t)#+int(num_tied_tot)

        #tf.print(mat_A, output_stream=sys.stdout)
        if Num != 0.0 and Den != 0.0:
            nb+=1
            resultat+=float(Num/Den)

        if resultat!=0:
            resultat = resultat/float(nb)
        return float(resultat)
    return cindex



def metric_cindex_weighted(time_horizon_dim,batch_size, y_val):
    time_eval= [a[0] for a in y_val] #this will make an error for sure
    event_eval= [a[0] for a in y_val]
    def cindex_weighted(y_true, y_pred):
        time=y_true[0]
        event=y_true[1]
        '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
        '''
        nb=0
        resultat=0.0
        ###############error 
        time_np=np.array([])
        event_np=np.array([])
        nb=0
        resultat=0.0
        tied=1e-2
        num_tied_tot=0
        num_tied_true=0
        for i in range(batch_size):
            time_np=np.append(time_np, time[i]) 
            event_np= np.append(event_np, event[i]) 

        kmf = KaplanMeierFitter()
        kmf.fit(time_np, event_observed=(event_np==0).astype(int))  # censoring prob = survival probability of event "censoring"
        G = np.asarray(kmf.survival_function_.reset_index()).transpose()
        G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)
        ################fin error
        for i in range (batch_size):
             #test
            tmp_idx= np.where(G[0,:]>= time_eval[i])[0]
            if len(tmp_idx)==0:
                W=(1/G[1,-1])**2
            else:
                W=(1/G[1, tmp_idx[0]])**2
            
            ######
            #results=tf.constant([])
            #a=tf.zeros([1,eval_time],dtype=tf.int32)
            #b=tf.ones([1,time_horizon_dim-eval_time],dtype=tf.int32)
            #c= tf.concat([a,b],1)
            a=tf.ones([1,time[i]],dtype=tf.int32)
            b=tf.zeros([1,time_horizon_dim-time[i]],dtype=tf.int32)
            c= tf.concat([a,b],1)
            c=tf.cast(c, dtype=tf.float32)
            d=y_pred*c
            pred_t=tf.math.reduce_sum(d,1, keepdims=True)
            
            ##########################
            time_equal= tf.math.equal(time[i], time)
            time_equal= tf.where(time_equal,1,0)
            time_equal=tf.cast(time_equal, dtype=tf.int32)
            event=tf.cast(event, dtype=tf.int32)
            time_equal= time_equal*event
            num_tied_tot+=tf.reduce_sum(time_equal[i+1:])
            pred_equal = tf.math.less(abs(pred_t[i]-pred_t), tied)
            pred_equal = tf.where(pred_equal, 1,0)
            ties = tf.transpose(pred_equal)*time_equal
            num_tied_true+=tf.reduce_sum(ties[:,i+1:])
            ##########################
            A=tf.math.greater(time, time[i])
            A=tf.where(A,1,0)
            Q=tf.math.greater(pred_t[i],pred_t)
            Q=tf.where(Q,1,0)
            #A[i, np.where(time[i] < time)] = 1
            #Q[i, np.where(pred_t[i] > pred_t)] = 1

            if (event[i]==1):
                N_t=tf.ones([1,batch_size])
            else:
                N_t=tf.zeros([1,batch_size])
                #N_t[i,:] = 1
                
            if i==0:
                mat_A=A
                mat_Q=Q
                mat_N_t=N_t
            else:
                mat_A=tf.concat([mat_A,A],0)
                mat_Q=tf.concat([mat_Q,Q],0)
                mat_N_t=tf.concat([mat_N_t,N_t],0)
        
        mat_A=tf.reshape(mat_A, [batch_size,batch_size])
        mat_Q=tf.reshape(mat_Q, [batch_size,batch_size])
        mat_N_t=tf.reshape(mat_N_t, [batch_size,batch_size])

        mat_A=tf.cast(mat_A,dtype=tf.float64)
        mat_Q=tf.cast(mat_Q,dtype=tf.float64)
        mat_N_t=tf.cast(mat_N_t,dtype=tf.float64)

        Num= tf.reduce_sum((mat_A*mat_N_t)*mat_Q)+(int(num_tied_true)/2)
        Den=tf.reduce_sum(mat_A*mat_N_t)#+int(num_tied_tot)

        #tf.print(mat_A, output_stream=sys.stdout)
        #Num  = np.sum(((A)*N_t)*Q)
        #Den  = np.sum((A)*N_t)
        if Num != 0.0 and Den != 0.0:
            nb+=1
            resultat+=float(Num/Den)

        if resultat!=0:# or resultat_tied!=0:
            #resultat = (resultat+resultat_tied)/(float(nb) + num_tied)
            resultat = resultat/float(nb)
        return float(resultat)
    return cindex_weighted

