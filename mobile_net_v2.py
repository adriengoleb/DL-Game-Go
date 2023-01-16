import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
import gc

import golois

planes = 31
moves = 361
N = 10000
epochs = 1000
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


input = keras.Input(shape=(19, 19, planes), name='board')
# x = layers.Conv2D(filters, 1, activation='relu', padding='same')(input)
x = layers.Conv2D(trunk, 1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(input)
x = layers.BatchNormalization()(x)
x1 = activations.sigmoid(x)
x = layers.Multiply()([x,x1])
for i in range (12):
    # Mobile Net Way
    m = layers.Conv2D(filters, (1,1), kernel_regularizer=regularizers.l2(1e-4), use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m1 = activations.sigmoid(m)
    m = layers.Multiply()([m,m1])
    m = layers.DepthwiseConv2D((3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4),use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    m1 = activations.sigmoid(m)
    m = layers.Multiply()([m,m1])
    #m = layers.Activation('relu')(m)
    m = layers.Conv2D(trunk, (1,1), kernel_regularizer=regularizers.l2(1e-4), use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    x = layers.Add()([m,x])
for i in range(0):
    # Residual Way
    x1 = layers.Conv2D(filters, 5, padding='same')(x)
    x1 = layers.BatchNormalization()(x1)
    x2 = layers.Conv2D(filters, 1, padding='same')(x)
    x  = layers.Add()([x1,x2])
    x  = layers.ReLU()(x)
    #activation='relu',
policy_head = layers.Conv2D(1, 1,  padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
value_head = layers.Flatten()(value_head)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

model = keras.Model(inputs=input, outputs=[policy_head, value_head])

model.summary ()

# model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=1e-3),
#               loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
#               loss_weights={'policy' : 1.0, 'value' : 1.0},
#               metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.92, beta_2=0.9991),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

for i in range (1, epochs + 1):
    print ('epoch ' + str (i))
    golois.getBatch (input_data, policy, value, end, groups, i * N)
    history = model.fit(input_data,
                        {'policy': policy, 'value': value},
                        epochs=1, batch_size=batch)
    if (i % 5 == 0):
        gc.collect ()
    if (i % 20 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model.evaluate (input_data,
                              [policy, value], verbose = 0, batch_size=batch)
        print ("val =", val)
        model.save ('Mobile_Net_v2.h5')