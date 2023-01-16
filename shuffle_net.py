import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
import gc
import os
import golois
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, Concatenate, Add, ReLU, BatchNormalization, AvgPool2D, MaxPool2D, GlobalAveragePooling2D, Reshape, Permute, Lambda, Flatten, Activation

import argparse


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

def bottleneck_block(tensor, expand=96, squeeze=16):
  x = gconv(tensor, channels=expand, groups=4)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = channel_shuffle(x, groups=4)
  x = DepthwiseConv2D(kernel_size=3, padding='same')(x)
  x = BatchNormalization()(x)
  x = gconv(x, channels=squeeze, groups=4)
  x = BatchNormalization()(x)
  x = Add()([tensor, x])
  output = ReLU()(x)
  return output

def main():

    parser = argparse.ArgumentParser( )
    parser.add_argument("-fn", "--file-name", type=str, default="shuffle_net",
                        help="file name, default = shuffle_net")
    parser.add_argument("-n", "--n-epochs", type=int, default= 400,
                        help="Number of epochs, default = 400.")

    args = parser.parse_args()

    file_name = args.file_name

    planes = 31
    moves = 361
    N = 10000
    epochs = args.n_epochs
    batch = 128
    # filters = 32
    filters = 48
    trunk = 24

    input_data = np.random.randint(2, size=(N, 19, 19, planes))
    input_data = input_data.astype ('float32')

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


    shuffle_blocks = 18
    with tf.device('/device:GPU:0'):
        input = keras.Input(shape=(19, 19, planes), name='board')
        x = Conv2D(trunk, 1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        for i in range (shuffle_blocks):
            x = bottleneck_block(x, filters, trunk)
        policy_head = Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
        policy_head = Flatten()(policy_head)
        policy_head = Activation('softmax', name='policy')(policy_head)
        value_head = GlobalAveragePooling2D()(x)
        value_head = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
        value_head = Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

        model = keras.Model(inputs=input, outputs=[policy_head, value_head])


        model.summary ()

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
                        loss_weights={'policy' : 1.0, 'value' : 1.0},
                        metrics={'policy': 'categorical_accuracy', 'value': 'mse'})
        val_list = []
        for i in range (1, epochs + 1):
            print ('epoch ' + str (i))
            golois.getBatch (input_data, policy, value, end, groups, i * N)
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
        model.save (file_name + '.h5')
        np.save(file_name+'_val.npy', np.array(val_list))

if __name__ == "__main__":
    main()