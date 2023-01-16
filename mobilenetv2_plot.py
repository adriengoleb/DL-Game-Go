import os
import math
import gc
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, Concatenate, Add, ReLU, BatchNormalization, AvgPool2D, MaxPool2D, GlobalAveragePooling2D, Reshape, Permute, Lambda, Flatten, Activation

import golois

planes = 31
moves = 361
N = 10000
epochs = 1500
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

plot_model(model, to_file='mobilenetv2_schema.png')