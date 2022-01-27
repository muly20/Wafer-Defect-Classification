import numpy as np
import tensorflow as tf
print(f"tensorflow version: {tf.__version__}")
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from sklearn import metrics

from utils_DCN import *
from DC_Layer_reduced import *

dataset = np.load('../Wafer_Defect/MixedWM38.npz')
dataset_x, dataset_y = dataset['arr_0'], dataset['arr_1']
m, n_H, n_W = dataset_x.shape

# reshape the input to have shape of image with 1 channel
dataset_x_cnn = np.expand_dims(dataset_x, axis=-1)

# split the dataset into train (80%), dev(10%) and test(10%) sets
shuffled_idx = np.asarray(list(range(dataset_x_cnn.shape[0])))
np.random.shuffle(shuffled_idx)
shuffled_x = dataset_x_cnn[shuffled_idx, :, :, :]
shuffled_y = dataset_y[shuffled_idx, :]

train_cut = int(shuffled_x.shape[0]*.80)

dev_cut = int(shuffled_x.shape[0]*.90)

train_x = shuffled_x[:train_cut, :, :, :]
train_y = shuffled_y[:train_cut, :]

dev_x = shuffled_x[train_cut:dev_cut, :, :, :]
dev_y = shuffled_y[train_cut:dev_cut, :]

test_x = shuffled_x[dev_cut:, :, :]
test_y = shuffled_y[dev_cut:, :]

def DCNN_model(input_shape, classes=8, trainable=True):
    """
    Deformable Convolution model with 4 Blocks of DC layers,
    followed by Global Avg Pooling and 1 FC output layer with sigmoid activation.
    :param input_shape: (52, 52, 1)
    :param classes: 8
    :return: keras model object
    """

    X_input = layers.Input(input_shape)

    # Block 1
    kernel_size_1 = 3
    strides_1 = 2
    offsets_1 = layers.Conv2D(name='Conv2D_offsets_1',
                              filters=2 * kernel_size_1 ** 2,
                              kernel_size=kernel_size_1,
                              strides=strides_1,
                              padding='same',
                              kernel_initializer='random_normal'
                              )(X_input)
    X = DefConvLayer_red(name='defconv_1',
                         filters=32,
                         kernel_size=kernel_size_1,
                         strides=strides_1
                         )(X_input, offsets_1)
    X = layers.BatchNormalization(axis=3, name='bn_1')(X)
    X = layers.Activation('relu')(X)

    # Block 2
    kernel_size_2 = 3
    strides_2 = 1
    offsets_2 = layers.Conv2D(name='Conv2D_offsets_2',
                              filters=2 * kernel_size_2 ** 2,
                              kernel_size=kernel_size_2,
                              strides=strides_2,
                              padding='same',
                              kernel_initializer='random_normal')(X)
    X = DefConvLayer_red(filters=64,
                         kernel_size=kernel_size_2,
                         strides=strides_2,
                         name='defconv_2')(X, offsets_2)
    X = layers.BatchNormalization(axis=3, name='bn_2')(X)
    X = layers.Activation('relu')(X)
    # X = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='MaxPool_2')(X)

    # Block 3
    kernel_size_3 = 3
    strides_3 = 2
    offsets_3 = layers.Conv2D(name='Conv2D_offsets_3',
                              filters=2 * kernel_size_3 ** 2,
                              kernel_size=kernel_size_3,
                              strides=strides_3,
                              padding='same',
                              kernel_initializer='random_normal')(X)
    X = DefConvLayer_red(filters=128,
                         kernel_size=kernel_size_3,
                         strides=strides_3,
                         name='defconv_3')(X, offsets_3)
    X = layers.BatchNormalization(axis=3, name='bn_3')(X)
    X = layers.Activation('relu')(X)

    # Block 4
    kernel_size_4 = 3
    strides_4 = 2
    offsets_4 = layers.Conv2D(name='Conv2D_offsets_4',
                              filters=2 * kernel_size_4 ** 2,
                              kernel_size=kernel_size_4,
                              strides=strides_4,
                              padding='same',
                              kernel_initializer='random_normal')(X)
    X = DefConvLayer_red(filters=128,
                         kernel_size=kernel_size_4,
                         strides=strides_4,
                         name='defconv_4')(X, offsets_4)
    X = layers.BatchNormalization(axis=3, name='bn_4')(X)
    X = layers.Activation('relu')(X)

    # Pooling Layer (instead of FC1) with number of units as the number of channels (256 above)
    X = layers.GlobalAvgPool2D(name='GlbAvgPool')(X)

    # FC output Layer
    X = layers.Dense(units=classes, activation='sigmoid', name='FC', trainable=trainable)(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)
    return model

model = DCNN_model(input_shape=(n_H, n_W, 1), classes=8, trainable=True)
print(model.summary())
plot_model(model)

# compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=.9) # default lr=0.001
metric = MultiLabel_Accuracy(name='multilabel_accuracy')
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metric])

# add checkpoints
checkpoint_filepath = '../Wafer_Defect/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               save_weights_only=False,
                                                               monitor='val_accuracy',
                                                               mode='max',
                                                               save_best_only=True)
model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model_history = model.fit(train_x, train_y,
                          epochs=20,
                          initial_epoch=10,
                          validation_data=(dev_x, dev_y),
                          batch_size=32,
                          callbacks=[model_checkpoint_callback, model_early_stopping])

# Evaluate on Test set
predictions = model.predict(test_x, batch_size=64, verbose=1)
predictions = np.where(predictions>.5, 1, 0)

metric.reset_state()
metric.update_state(test_y, predictions)
print(f"Test set accuracy: {metric.result().numpy()}")

f_score = metrics.f1_score(test_y, predictions, average=None)
print(f"confusion metrices:\n{metrics.multilabel_confusion_matrix(test_y, predictions)}")
print(f"f-score per class:\n{f1_score}")
print(f"average f1 score: {np.average(f1_score)}")

# Evaluating Accuracy by Fault Type
# defining Pattern Type Indices
type = {}
# Single-Type Fault
type['C1'] = np.arange(33866, 34866)
type['C2'] = np.arange(12000, 13000)
type['C3'] = np.arange(24000, 25000)
type['C4'] = np.arange(25000, 26000)
type['C5'] = np.arange(26000, 27000)
type['C6'] = np.arange(32000, 33000)
type['C7'] = np.arange(33000, 33866)
type['C8'] = np.arange(37015, 38015)
type['C9'] = np.arange(34866, 35015)

# Two-Types Fault
type['C10'] = np.arange(2000, 3000)
type['C11'] = np.arange(4000, 5000)
type['C12'] = np.arange(10000, 11000)
type['C13'] = np.arange(11000, 12000)
type['C14'] = np.arange(14000, 15000)
type['C15'] = np.arange(16000, 17000)
type['C16'] = np.arange(22000, 23000)
type['C17'] = np.arange(23000, 24000)
type['C18'] = np.arange(28000, 29000)
type['C19'] = np.arange(35015, 36015)
type['C20'] = np.arange(30000, 31000)
type['C21'] = np.arange(36015, 37015)
type['C22'] = np.arange(31000, 32000)

# Three-Types Fault
type['C23'] = np.arange(6000, 7000)
type['C24'] = np.arange(0, 2000)
type['C25'] = np.arange(8000, 9000)
type['C26'] = np.arange(3000, 4000)
type['C27'] = np.arange(9000, 10000)
type['C28'] = np.arange(18000, 19000)
type['C29'] = np.arange(13000, 14000)
type['C30'] = np.arange(20000, 21000)
type['C31'] = np.arange(15000, 16000)
type['C32'] = np.arange(21000, 22000)
type['C33'] = np.arange(27000, 28000)
type['C34'] = np.arange(29000, 30000)
type['C35'] = np.arange(5000, 6000)
type['C36'] = np.arange(7000, 8000)
type['C37'] = np.arange(17000, 18000)
type['C38'] = np.arange(19000, 20000)

acc_list = []

for i in range(1, 39):
  t = 'C'+str(i)
  acc = type_eval(type[t])
  acc_list.append(acc)

