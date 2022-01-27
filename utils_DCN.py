import tensorflow as tf
import numpy as np

def type_eval(indices):
  input = dataset_x_cnn[indices, ...]
  pred = model.predict(input, batch_size=32, verbose=1)
  pred = np.where(pred>0.5, 1, 0)

  metric.reset_state()
  metric.update_state(dataset_y[indices, :], pred)

  return metric.result().numpy()

class MultiLabel_Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='multilabel_accuracy', threshold=0.5):
        super(MultiLabel_Accuracy, self).__init__(name=name)
        self.accuracy = self.add_weight(name='accuracy', initializer='zeros')
        self.threshold = threshold
        self.count = self.add_weight(initializer='zeros')

    def result(self):
        return tf.math.divide(self.accuracy, self.count)

    def reset_state(self):
        self.accuracy.assign(0)
        self.count.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.count.assign_add(y_true.shape[0])
        pred = tf.where(y_pred > self.threshold, 1, 0)

        tmp = tf.cast(y_true == pred, dtype='float32')
        tmp = tf.reduce_sum(tmp, axis=-1)
        # hard-coded 8 labels total
        tmp = tf.where(tmp==8, 1, 0)
        tmp = tf.reduce_sum(tmp)

        self.accuracy.assign_add(tf.cast(tmp, dtype='float32'))