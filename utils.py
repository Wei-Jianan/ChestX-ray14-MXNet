import logging
import sys

import mxnet as mx
import os

from sklearn import metrics

import numpy as np
from mxnet import nd, gluon, autograd

from time import time

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))



def set_logging_level(level):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=level, stream=sys.stdout)


def _get_batch(batch, ctx):
    """return features and labels on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        features = batch.data[0]
        labels = batch.label[0]
    else:
        features, labels = batch
    return (gluon.utils.split_and_load(features, ctx),
            gluon.utils.split_and_load(labels, ctx),
            features.shape[0])


def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes


def evaluate_loss(data_iter, net, loss_fn=gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False), ctx=[mx.cpu()]):
    if isinstance(data_iter, mx.io.MXDataIter):
        data_iter.reset()
    loss_sum = 0
    for batch in data_iter:
        #         print('X in evluating:', X         )
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            output = net(X)
            logging.info('output shape:\t{}'.format(output.shape))
            loss = loss_fn(output, y)
            logging.info('loss in batch:\t{}'.format(loss))
            loss_sum += loss.mean().asscalar()
    return loss_sum / len(data_iter)


def evaluate_AUC(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    if isinstance(data_iter, mx.io.MXDataIter):
        data_iter.reset()

    outputs_in_batches = []
    labels_in_batches = []
    _first_num_class_left = 6  # for debugging
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y_hat = net(X)
            y_hat = y_hat[:, :_first_num_class_left]  # for debugging
            outputs_in_batches.append(y_hat)
            # logging.debug(y_hat[:5])
            y = y[:, :_first_num_class_left]  # for debugging
            labels_in_batches.append(y)
            # logging.debug(y[:5])

        nd.waitall()

    AUC = _calculate_AUC(outputs_in_batches, labels_in_batches)

    #     print('increase mem 2:\t', get_mem() - mem)
    return AUC


def _calculate_AUC(outputs_in_batches, labels_in_batches):
    num_samples = sum([len(output) for output in outputs_in_batches])
    num_classes = labels_in_batches[0].shape[1]
    logging.info('num_samplesL\t{}'.format(num_samples))
    logging.info('num_classes:\t{}'.format(num_classes))

    all_outputs = np.zeros((num_samples, num_classes), dtype='float32')
    all_labels = np.zeros((num_samples, num_classes), dtype='float32')

    to_update = 0
    start_copy = time()
    for output, label in zip(outputs_in_batches, labels_in_batches):
        all_outputs[to_update: to_update + len(output)] = output.asnumpy()
        all_labels[to_update: to_update + len(output)] = label.asnumpy()
        to_update += len(output)
    logging.debug(all_outputs)
    logging.info('take {} s to copy data from gpu to cpu when evaluating'.format(time() - start_copy))

    AUC = metrics.roc_auc_score(y_true=all_labels, y_score=all_outputs, average=None)
    logging.info('AUC :\t{}'.format(AUC))

    return AUC


def train_parallel(train_iter, test_iter, net, loss_fn, trainer, ctx, num_epochs, print_batches=None):
    """Train and evaluate a model."""
    print("training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(1, num_epochs + 1):
        if isinstance(train_iter, mx.io.MXDataIter):
            train_iter.reset()
        start = time()

        train_loss_sum = 0
        outputs_in_batches = []
        labels_in_batches = []
        _first_num_class_left = 6  # for debugging

        for batch in train_iter:
            features, labels, batch_size = _get_batch(batch, ctx)
            losses = []
            for X, y in zip(features, labels):
                with autograd.record(train_mode=True):
                    y_hat = net(X)
                    loss = loss_fn(y_hat, y)
                loss.backward()
                losses.append(loss)

                # y_hat = y_hat[:, :_first_num_class_left]  # for debugging
                outputs_in_batches.append(y_hat)  # this will take about 10m more space in GPU memory.
                # logging.info('y_hat.shape:{}'.format(y_hat.shape))
                # logging.debug(y_hat[:5])
                # y = y[:, :_first_num_class_left]  # for debugging
                labels_in_batches.append(y)
                # logging.info('y.shape:\t{}'.format(y.shape))
                # logging.debug(y[:5])

            trainer.step(batch_size)
            train_loss_sum += sum([l.sum().asscalar() for l in losses]) / batch_size

            nd.waitall()

        train_AUC = _calculate_AUC(outputs_in_batches, labels_in_batches)
        test_AUC = evaluate_AUC(test_iter, net, ctx)

        print("epoch %d, loss %.4f, time %.1f sec" % (
            epoch, train_loss_sum / len(train_iter), time() - start
        ))
        print('train AUC:\n', train_AUC)
        print('test AUC:\n', test_AUC)


if __name__ == "__main__":
    # data_dir = upzip_and_delete()
    pass
