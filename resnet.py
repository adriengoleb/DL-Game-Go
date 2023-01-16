import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
import gc
import os
import golois


import argparse


def main():

    parser = argparse.ArgumentParser( )
    parser.add_argument("-fn", "--file-name", type=str, default="resnet",
                        help="file name, default = resnet")
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


    with tf.device('/device:GPU:0'):
        
        input = keras.Input(shape=(19, 19, planes), name='board')
        #block1
        x1 = layers.Conv2D(32, 3, activation='relu', padding='same')(input)
        x1 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x1)
        x1 = layers.Conv2D(32, 3, activation='relu', padding='same')(x1)
        x1 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x1)
        #block1 end
        #block2
        x2 = layers.Conv2D(32, 3, activation='relu', padding='same')(x1)
        x2 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x2)
        x2 = layers.add([x1,x2])
        x2 = layers.Conv2D(32, 3, activation='relu', padding='same')(x2)
        x2 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x2)
        #block2 end
        #block 3
        x3 = layers.Conv2D(32, 3, activation='relu', padding='same')(x2)
        x3 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x3)
        x3 = layers.add([x2,x3])
        x3 = layers.Conv2D(32, 3, activation='relu', padding='same')(x3)
        x3 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x3) 
        #block3 end
        #block 4
        x4 = layers.Conv2D(32, 3, activation='relu', padding='same')(x3)
        x4 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x4)
        x4 = layers.add([x3,x4])
        x4 = layers.Conv2D(32, 3, activation='relu', padding='same')(x4)
        x4 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x4) 
        # #block4 end
        #block5
        x5 = layers.Conv2D(32, 3, activation='relu', padding='same')(x4)
        x5 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x5)
        x5 = layers.add([x4,x5])
        x5 = layers.Conv2D(32, 3, activation='relu', padding='same')(x5)
        x5 = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x5) 
        #block5 end
        policy_head = layers.Conv2D(1, 1, activation='swish', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x4)
        policy_head = layers.Flatten()(policy_head)
        policy_head = layers.Activation('softmax', name='policy')(policy_head)
        value_head = layers.GlobalAveragePooling2D()(x4)
        value_head = layers.Dense(50, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(value_head)
        value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

        model = keras.Model(inputs=input, outputs=[policy_head, value_head])

        model.summary ()

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
                        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
                        loss_weights={'policy' : 1.0, 'value' : 1.0},
                        metrics={'policy': 'categorical_accuracy', 'value': 'mse'})
        val_list = []
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
        model.save (file_name + '.h5')
        np.save(file_name+'_val.npy', np.array(val_list))

if __name__ == "__main__":
    main()