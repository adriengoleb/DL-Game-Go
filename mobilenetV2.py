import os
import gc
import math
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, Concatenate, Add, ReLU, BatchNormalization, AvgPool2D, MaxPool2D, GlobalAveragePooling2D, Reshape, Permute, Lambda, Flatten, Activation

import golois

planes = 31
moves = 361
N = 10000
epochs = 1000
# epochs = 20
batch=64 
filters = 64
trunk = filters

input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')

print(input_data.ndim)
print(input_data.shape)

policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)

value = np.random.randint(2, size=(N,))
value = value.astype ('float32')

end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')

groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')

print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)

def bottleneck_block_mobilenet(x, expand=512, squeeze=128):
# def bottleneck_block_mobilenet(x, expand=96, squeeze=16):
    m = layers.Conv2D(expand, (1,1), kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(x)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)
    m = layers.DepthwiseConv2D((3,3), padding='same', kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)
    m = layers.Conv2D(squeeze, (1,1), kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(m)
    m = layers.BatchNormalization()(m)
    return layers.Add()([m, x])

def bottleneck_block_shufflenet(tensor, expand=512, squeeze=128):
# def bottleneck_block_shufflenet(tensor, expand, squeeze=64):
    x = gconv(tensor, channels=expand, groups=4)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = channel_shuffle(x, groups=4)
    x = DepthwiseConv2D(kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.0001), use_bias = False)(x)
    x = BatchNormalization()(x)

    x = gconv(x, channels=squeeze, groups=4)
    x = BatchNormalization()(x)

    x = Add()([tensor, x])
    output = ReLU()(x)
    return output

def gconv(tensor, channels, groups):
    input_ch = tensor.get_shape().as_list()[-1]
    group_ch = input_ch // groups
    output_ch = channels // groups
    groups_list = []

    for i in range(groups):
        group_tensor = tensor[:, :, :, i * group_ch: (i+1) * group_ch]
        group_tensor = Conv2D(output_ch, 1)(group_tensor)
        groups_list.append(group_tensor)

    output = Concatenate()(groups_list)
    return output

def gconv(tensor, channels, groups):
    input_ch = tensor.get_shape().as_list()[-1]
    group_ch = input_ch // groups
    output_ch = channels // groups
    groups_list = []

    for i in range(groups):
        group_tensor = tensor[:, :, :, i * group_ch: (i+1) * group_ch]
        group_tensor = Conv2D(output_ch, 1)(group_tensor)
        groups_list.append(group_tensor)

    output = Concatenate()(groups_list)
    return output

def channel_shuffle(x, groups):  
    _, width, height, channels = x.get_shape().as_list()
    group_ch = channels // groups

    x = Reshape([width, height, group_ch, groups])(x)
    x = Permute([1, 2, 4, 3])(x)
    x = Reshape([width, height, channels])(x)
    return x


# def getModel ():
#    input = keras.Input(shape=(19, 19, planes), name='board')
#    x = Conv2D(trunk, 1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(input)
#    x = BatchNormalization()(x)
#    x = ReLU()(x)
#    for i in range (blocks):
#        x = bottleneck_block_shufflenet (x, filters, trunk)
#    policy_head = Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
#    policy_head = Flatten()(policy_head)
#    policy_head = Activation('softmax', name='policy')(policy_head)
#    value_head = GlobalAveragePooling2D()(x)
#    value_head = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
#    value_head = Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)
        
#    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
#    return model


input = keras.Input(shape=(19, 19, planes), name='board')

# This one is the mobile net (Thomas' version)
# x = layers.Conv2D(trunk, 1, activation='relu', padding='same')(input)
# for i in range (blocks):
#    x = bottleneck_block_mobilenet(x, filters, trunk)
# This one is the mobile net (Antonio's version)
x = layers.Conv2D(trunk, 1, padding='same',kernel_regularizer=regularizers.l2(0.0001))(input)
x_bifurcation = BatchNormalization()(x)

blocks = 33 # 33
filters = 198 # 200
x = ReLU()(x_bifurcation)
for i in range (blocks):
    x = bottleneck_block_mobilenet(x, filters, trunk)

# In series
# for i in range (blocks):
#    x = bottleneck_block_shufflenet(x, filters, trunk)


# In parallel
# y = ReLU()(x_bifurcation)
# for i in range (blocks):
#    y = bottleneck_block_shufflenet(y, filters, trunk)
# concatenate
# x = layers.concatenate([x, y], axis=-1)

    
# fully convolutional, no dense layer
policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)

value_head = layers.GlobalAveragePooling2D()(x)
value_head = layers.Dense(50, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value',kernel_regularizer=regularizers.l2(0.0001))(value_head)

model = keras.Model(inputs=input, outputs=[policy_head, value_head])

model.summary ()

# 2nd Train momentum 0.8; 2nd Train momentum 0.9
# model.compile(optimizer=keras.optimizers.SGD(lr=0.000001, momentum=0.9,nesterov=False),
# 2nd Train
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              # 2nd Train
              loss_weights={'policy' : 2.0, 'value' : 1.0},
              # loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

for i in range (1, epochs + 1):
    print ('epoch ' + str (i))
    golois.getBatch(input_data, policy, value, end, groups, i * N)
    history = model.fit(input_data,
                        {'policy': policy, 'value': value}, 
                        epochs=1, batch_size=batch)
    if (i % 5 == 0):
        gc.collect ()
    if (i % 10 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model.evaluate (input_data,
                              [policy, value], verbose = 0, batch_size=batch)
        print ("val =", val)
        model.save("mobilenetV2.h5")

