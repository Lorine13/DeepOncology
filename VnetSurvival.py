import tensorflow as tf
from networks.Layers import convolution, down_convolution, up_convolution, get_num_channels
from networks.Vnet import *
import numpy as np


def dropout(x, keep_prob):
    # tf.keras.layers.Dropout(1.0 - keep_prob)(x)
    return tf.keras.layers.SpatialDropout3D(1.0 - keep_prob)(x)


def convolution_block(layer_input, num_convolutions, kernel_size, keep_prob, activation_fn):
    x = layer_input
    n_channels = get_num_channels(x)
    for i in range(num_convolutions):
        x = convolution(x, n_channels, kernel_size=kernel_size)
        # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = activation_fn(x)
        if keep_prob < 1.0:
            x = dropout(x, keep_prob)

    x = x + layer_input
    return x


def convolution_block_2(layer_input, fine_grained_features, num_convolutions, kernel_size, keep_prob, activation_fn):

    x = tf.keras.layers.Concatenate()([layer_input, fine_grained_features])
    n_channels = get_num_channels(layer_input)
    for i in range(num_convolutions):
        x = convolution(x, n_channels, kernel_size=kernel_size)
        # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = activation_fn(x)
        if keep_prob < 1.0:
            x = dropout(x, keep_prob)

    # layer_input = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(layer_input)
    x = x + layer_input
    return x



def get_logLikelihood_LOSS(time, event, time_horizon_dim, y_pred):
    '''
    for uncensored : log of the probabilitie predicted to experience the event at the true time t
    for censored : log of the sum of probabilities predicted to experience the event after the true censoring time t
    '''

    #mask for the log-likelihood loss
    #mask size is [N, time_horizon_dim]
    #    if not censored : one element = 1 (0 elsewhere)
    #    if censored     : fill elements with 1 after the censoring time (for all events)
    mask = np.zeros([len(time), time_horizon_dim]) 
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

    return loss

def get_ranking_LOSS(time, event, time_horizon_dim, y_pred):
    '''
    for pairs acceptables (careful with censored events):
    loss  function η(P(x),P(y)) = exp(−(P(x)−P(y))/σ)
    where P(x) is the sum of probabilities for x to experience the event on time t <= tx -- (true time of x) --
    and P(y) is the sum of probabilities for y to experience the event on time t <= tx
    translated to : a patient who dies at time s should have a higher risk at time s than a patient who survived longer than s
    '''
    #    mask is required calculate the ranking loss (for pair-wise comparision)
    #    mask size is [N, time_horizon_dim].
    #         1's from start to the event time(inclusive)
    mask = np.zeros([len(time), time_horizon_dim])
    for i in range(len(time)):
        t = int(time[i]) # censoring/event time
        mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    
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

    loss = tf.reduce_sum(tf.reduce_mean(T2 * tf.exp(-R2/sigma),1, keepdims=True))

    return loss

def get_loss_survival(time_horizon_dim):
    ''' 
    time_horizon_dim : output dimension of the output layer of the model
    loss_survival : returns the loss of the model (log_likelihood loss + ranking loss)
    '''
    def loss_survival(y_true, y_pred):
        time=y_true.numpy()[0]
        event=y_true.numpy()[1]

        loss_logLikelihood= get_logLikelihood_LOSS(time, event, time_horizon_dim, y_pred)
        loss_ranking= get_ranking_LOSS(time, event, time_horizon_dim, y_pred)

        loss= loss_logLikelihood + loss_ranking
        
        return loss

    return loss_survival

def metric_td_c_index(time_horizon_dim):
    
    def td_c_index(y_true, y_pred):
        time=y_true.numpy()[0]
        event=y_true.numpy()[1]
        y_pred=y_pred.numpy()
        '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
        '''
        for eval_time in range (time_horizon_dim):
            pred_t = np.sum(y_pred[:,:(eval_time+1)], axis=1)
        
            N = len(y_pred)
            A = np.zeros((N,N))
            Q = np.zeros((N,N))
            N_t = np.zeros((N,N))
            Num = 0
            Den = 0
            results=np.array([])
            for i in range(N):
                A[i, np.where(time[i] < time)] = 1
                Q[i, np.where(pred_t[i] > pred_t)] = 1

                if (time[i]<=eval_time and event[i]==1):
                    N_t[i,:] = 1
            Num  = np.sum(((A)*N_t)*Q)
            Den  = np.sum((A)*N_t)
            if Num == 0 and Den == 0:
                results = -1 # not able to compute c-index!
            else:
                results.append(float(Num/Den))
        
        result = np.mean(results)
        print( 'average c-index = ' + str('%.4f' %(result)))
        return result
    return td_c_index




class VnetSurvival(object):
    """
    Implements VNet architecture https://arxiv.org/abs/1606.04797
    """
    def __init__(self,
                 image_shape,
                 in_channels,
                 out_channels,
                 time_horizon,
                 channels_last=True,
                 keep_prob=1.0,
                 keep_prob_last_layer=1.0,
                 kernel_size=(5, 5, 5),
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 activation=tf.keras.layers.PReLU(),
                 activation_last_layer='softmax'):

        """
        :param image_shape: Shape of the input image
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param channels_last: bool, set to True for channels last format
        :param kernel_size: Size of the convolutional patch
        :param keep_prob: Dropout keep probability in the conv layer,
                            set to 1.0 if not training or if no dropout is desired.
        :param keep_prob_last_layer: Dropout keep probability in the last conv layer,
                                    set to 1.0 if not training or if no dropout is desired.
        :param num_channels: The number of output channels in the first level, this will be doubled every level.
        :param num_levels: The number of levels in the network. Default is 4 as in the paper.
        :param num_convolutions: An array with the number of convolutions at each level.
        :param bottom_convolutions: The number of convolutions at the bottom level of the network.
        :param activation: The activation function.
        :param activation_last_layer: The activation function used in the last layer of the cnn.
                                      Set to None to return logits.
        """
        self.image_shape = image_shape
        assert len(image_shape) == 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_horizon = time_horizon
        self.channels_last = channels_last
        assert channels_last  # channels_last=False is not supported
        self.kernel_size = kernel_size
        self.keep_prob = keep_prob
        self.keep_prob_last_layer = keep_prob_last_layer
        self.num_channels = num_channels
        assert num_levels == len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.activation_fn = activation
        if isinstance(self.activation_fn, str):
            self.activation_fn = self.activation_fn.lower()
            self.activation_fn = tf.keras.layers.PReLU() if self.activation_fn == 'prelu' \
                else tf.keras.activations.get(self.activation_fn)
        self.activation_last_layer = activation_last_layer.lower() if isinstance(activation_last_layer, str) \
            else activation_last_layer



    def build_network(self, input_):

        #self.loss_log_likelihood()
        #self.cause_specific_ranking_loss()

        x = input_
        keep_prob = self.keep_prob

        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input
        # channel
        input_channels = int(x.get_shape()[-1])
        if input_channels == 1:
            x = tf.keras.backend.tile(x, [1, 1, 1, 1, self.num_channels])
        else:
            x = convolution(x, self.num_channels, kernel_size=self.kernel_size)
            # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
            x = self.activation_fn(x)

        forwards = list()
        for l in range(self.num_levels):
            x = convolution_block(x, self.num_convolutions[l], self.kernel_size, keep_prob, activation_fn=self.activation_fn)
            forwards.append(x)
            x = down_convolution(x, factor=2, kernel_size=(2, 2, 2))
            # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
            x = self.activation_fn(x)

        x = convolution_block(x, self.bottom_convolutions, self.kernel_size, keep_prob, activation_fn=self.activation_fn)

        # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        if self.keep_prob_last_layer < 1.0:
            x = tf.keras.layers.Dropout(1.0 - self.keep_prob_last_layer)(x)
        x = convolution(x, self.out_channels)

        x = tf.keras.layers.Flatten()(x)
        
        x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(x)
        x = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(x)
        logits = tf.keras.layers.Dense(self.time_horizon, activation='relu', name='dense_3')(x)

        return logits

    def create_model(self):
        input_shape = tuple(list(self.image_shape) + [self.in_channels])
        input_ = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name="input")
        logits = self.build_network(input_)        
        output_ = tf.keras.layers.Softmax(name='output')(logits)
        model = tf.keras.models.Model(input_, output_, name='VNetSurvival')
        return model

