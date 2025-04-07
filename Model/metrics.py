import tensorflow as tf
from tensorflow.python import keras

class PrecisionMetric(keras.models.metrics_module.Metric):
    def __init__(self, name='precision', **kwargs):
        super(PrecisionMetric, self).__init__(name=name, **kwargs)
        self.intersect = self.add_weight(name='intersect', initializer='zeros')
        self.total_pixel_pred = self.add_weight(name='total_pixel_pred', initializer='zeros')

    def update_state(self, groundtruth_mask, pred_mask, sample_weight=None):
        groundtruth_mask = tf.cast(groundtruth_mask, tf.float32)
        pred_mask = tf.cast(pred_mask, tf.float32)
        intersect = tf.reduce_sum(pred_mask * groundtruth_mask)
        total_pixel_pred = tf.reduce_sum(pred_mask)
        self.intersect.assign_add(intersect)
        self.total_pixel_pred.assign_add(total_pixel_pred)

    def result(self):
        precision = self.intersect / (self.total_pixel_pred + keras.models.backend.epsilon())
        return tf.round(precision * 1000) / 1000

    def reset_states(self):
        self.intersect.assign(0)
        self.total_pixel_pred.assign(0)

class RecallMetric(keras.models.metrics_module.Metric):
    def __init__(self, name='recall', **kwargs):
        super(RecallMetric, self).__init__(name=name, **kwargs)
        self.intersect = self.add_weight(name='intersect', initializer='zeros')
        self.total_pixel_truth = self.add_weight(name='total_pixel_truth', initializer='zeros')

    def update_state(self, groundtruth_mask, pred_mask, sample_weight=None):
        groundtruth_mask = tf.cast(groundtruth_mask, tf.float32)
        pred_mask = tf.cast(pred_mask, tf.float32)
        intersect = tf.reduce_sum(pred_mask * groundtruth_mask)
        total_pixel_truth = tf.reduce_sum(groundtruth_mask)
        self.intersect.assign_add(intersect)
        self.total_pixel_truth.assign_add(total_pixel_truth)

    def result(self):
        recall = self.intersect / (self.total_pixel_truth + keras.models.backend.epsilon())
        return tf.round(recall * 1000) / 1000

    def reset_states(self):
        self.intersect.assign(0)
        self.total_pixel_truth.assign(0)

class IoUMetric(keras.models.metrics_module.Metric):
    def __init__(self, name='iou', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.intersect = self.add_weight(name='intersect', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersect = tf.reduce_sum(y_pred * y_true)
        union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) - intersect

        self.intersect.assign_add(intersect)
        self.union.assign_add(union)

    def result(self):
        iou = self.intersect / (self.union + keras.models.backend.epsilon())
        return tf.round(iou * 1000) / 1000

    def reset_states(self):
        self.intersect.assign(0)
        self.union.assign(0)