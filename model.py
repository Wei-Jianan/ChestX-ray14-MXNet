from mxnet import gluon, init, nd
from mxnet.gluon import nn


def get_pretrained_resnet50(ctx):
    pretrained_net = gluon.model_zoo.vision.resnet50_v1(pretrained=True, ctx=ctx)
    # temp_data = nd.random_normal(shape=(1, 4096))
    output_layer = nn.Dense(14, weight_initializer=init.Xavier())
    output_layer.initialize(ctx=ctx)

    tuning_net = nn.HybridSequential()
    with tuning_net.name_scope():
        tuning_net.add(pretrained_net.features,
                       output_layer)

    return tuning_net
