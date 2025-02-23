from typing import Callable, Iterable, List

import keras
import tensorflow as tf
from keras import callbacks, metrics
from tensorflow.python.keras.callbacks import CallbackList, EarlyStopping


class AggregatedMetrics:
    def __init__(self, l: Iterable[metrics.Metric]):
        self.metrics = sorted(l, key=lambda metric: metric.name)

    def __str__(self):
        def to_str(metric: metrics.Metric):
            return f'{metric.name}: {metric.result():.4f}'

        return ','.join(map(lambda m: to_str(m), self.metrics))

    @staticmethod
    def which(metric: metrics.Metric, w: str) -> bool:
        if w == '*':
            return True
        elif w == 'train' and not metric.name.startswith('val'):
            return True
        elif w == 'val' and metric.name.startswith('val'):
            return True
        return False

    def update(self, y_batch, y_pred, which: str='train'):
        for metric in self.metrics:
            if self.which(metric, which):
                metric.update_state(y_batch, y_pred)

    def reset(self, which: str='train'):
        for metric in self.metrics:
            if self.which(metric, which):
                metric.reset_state()


@tf.function
def train_step(x, y, model: keras.Model, loss_fn: keras.Loss, optimizer: keras.Optimizer, aggr: AggregatedMetrics):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        aggr.update(y, logits)
        return loss_value


@tf.function
def val_step(x, y, model: keras.Model, loss_fn: keras.Loss, aggr: AggregatedMetrics):
    val_logits = model(x, training=False)
    val_loss = loss_fn(y, val_logits)
    aggr.update(y, val_logits, which='val')
    return val_loss


def do(callbacks_fn: Iterable[callbacks.Callback], action: Callable[[callbacks.Callback], None]):
    for callback in callbacks_fn:
        action(callback)

def try_get_early_stopping(callbacks_fn: Iterable[callbacks.Callback]) -> EarlyStopping | None:
    try:
        return next(filter(lambda callback: isinstance(callbacks_fn, EarlyStopping), callbacks_fn))
    except StopIteration:
        return None

def training_loop(
        x_train, y_train, x_val, y_val,
        model: keras.Model,
        epochs: int,
        batch_size: int,
        metric_list: List[metrics.Metric],
        optimizer: keras.Optimizer,
        callbacks_fn: List[callbacks.Callback],
        loss_fn: keras.Loss,
        preprocessing_fn,
        verbose: bool=True,
):
    aggregate_metrics = AggregatedMetrics(metric_list)
    callback_list = CallbackList(add_history=True, model=model)
    callbacks_fn = list(callbacks_fn)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    early_stopping = try_get_early_stopping(callbacks_fn)
    callback_list.on_train_begin()
    do(callbacks_fn, lambda c: c.on_train_begin())
    for epoch in range(epochs):
        if verbose:
            print(f'Start of epoch {epoch}')

        callback_list.on_epoch_begin(epoch)
        do(callbacks_fn, lambda c: c.on_epoch_begin(epoch))
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            callback_list.on_batch_begin(step)
            callback_list.on_train_batch_begin(step)
            do(callbacks_fn, lambda c: c.on_train_batch_begin(step))

            x_batch_train, y_batch_train = preprocessing_fn(x_batch_train, y_batch_train)
            loss_value = train_step(x_batch_train, y_batch_train, model, loss_fn, optimizer, aggregate_metrics)

            callback_list.on_train_batch_end(step)
            do(callbacks_fn, lambda c: c.on_train_batch_end(step, {'loss': loss_value}))
            if verbose and step % 200 == 0:
                print(f'Training loss (for one batch) at step {step}: {float(loss_value):.4f}')
                print(f'Seen so far: {((step + 1) * batch_size)} samples')


        val_loss = val_step(x_val, y_val, model, loss_fn, aggregate_metrics)
        if early_stopping is not None and early_stopping.stopped_epoch > 0:
            print(f'Early stopping stopped training, restoring weights at epoch {early_stopping.stopped_epoch}')
            break
        if verbose:
            print(str(aggregate_metrics))

        aggregate_metrics.reset(which='*')
        callback_list.on_epoch_end(epoch)
        do(callbacks_fn, lambda c: c.on_epoch_end(epoch, {'val_loss': val_loss}))

    callback_list.on_train_end()
    do(callbacks_fn, lambda c: c.on_train_end())
    return get_history(callback_list)


def get_history(callback_list: CallbackList) -> callbacks.History | None:
    for callback in callback_list:
        if isinstance(callback, callbacks.History):
            return callback
    return None