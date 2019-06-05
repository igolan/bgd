import torch.nn as nn
import math
import torch


def init_model(model, **kwargs):
    conv_type = kwargs.get("conv_type", None)
    if conv_type == "def":
        conv_type = None
    bias_type = kwargs.get("bias_type", None)
    if bias_type == "def":
        bias_type = None
    mode = kwargs.get("mode", None)
    nonlinearity = kwargs.get("nonlinearity", None)
    bn_init = kwargs.get("bn_init", None)
    if bn_init == "def":
        bn_init = None
    init_linear = kwargs.get("init_linear", False)
    logger = kwargs.get("logger", None)

    assert (logger is not None)
    assert (conv_type in {"he", "xavier", None})
    assert (mode in {"fan_out", "fan_in", None})
    assert (nonlinearity in {"relu", None})
    assert (bn_init in {"01", "11", "uniformweight", None})

    if conv_type is not None:
        if conv_type == "he":
            init_conv_he(model, mode, nonlinearity, logger)
        if conv_type == "xavier":
            init_conv_xavier(model, mode, nonlinearity, logger)
            init_lin_xavier(model, logger)

    if bias_type is not None:
        if bias_type == "xavier":
            init_bias_xavier(model, mode, nonlinearity, logger)
            init_bias_lin_xavier(model, logger)
        if str(bias_type) == "0":
            init_bias_zero(model, mode, nonlinearity, logger)
            init_bias_lin_zero(model, logger)

    if bn_init is not None:
        if bn_init == "01":
            init_bn_01(model, logger=logger)
        elif bn_init == "11":
            init_bn_11(model, logger=logger)
        elif bn_init == "uniformweight":
            init_bn_uniformweight(model, logger=logger)
    if init_linear:
        init_lin(model, logger)


def init_conv_xavier(model, mode='fan_out', nonlinearity='relu', logger=None):
    layers_initialized = 0
    a = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            layers_initialized += 1
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.normal_(0, math.sqrt(2)/math.sqrt(1+9*m.bias.data.shape[0]))
    logger.info("Initialized " + str(layers_initialized) + " Conv2d layers using nn.init.xavier_normal_")


def init_bias_xavier(model, mode='fan_out', nonlinearity='relu', logger=None):
    layers_initialized = 0
    a = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                layers_initialized += 1
                m.bias.data.normal_(0, math.sqrt(2)/math.sqrt(1+9*m.bias.data.shape[0]))
    logger.info("Initialized " + str(layers_initialized) + \
                " bias conv2d layers using nn.init.xavier.noraml_")


def init_lin_xavier(model, logger=None):
    layers_initialized = 0
    a = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            layers_initialized += 1
            torch.nn.init.xavier_normal_(m.weight.data)
    logger.info("Initialized " + str(layers_initialized) + " linear layers using xavier")


def init_bias_lin_xavier(model, logger=None):
    layers_initialized = 0
    a = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                layers_initialized += 1
                m.bias.data.normal_(0, math.sqrt(2)/math.sqrt(1+m.bias.data.shape[0]))

    logger.info("Initialized " + str(layers_initialized) + " bias linear layers using xavier")


def init_bias_lin_zero(model, logger=None):
    layers_initialized = 0
    a = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                layers_initialized += 1
                m.bias.data.zero_()

    logger.info("Initialized " + str(layers_initialized) + " bias linear layers using 0")


def init_bias_zero(model, mode='fan_out', nonlinearity='relu', logger=None):
    layers_initialized = 0
    a = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                layers_initialized += 1
                m.bias.data.zero_()
    logger.info("Initialized " + str(layers_initialized) + \
                " bias conv2d layers using nn.init.zero")


def init_conv_he(model, mode='fan_out', nonlinearity='relu', logger=None):
    layers_initialized = 0
    a = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            layers_initialized += 1
            nn.init.kaiming_normal(m.weight.data, a=a, mode=mode)
    logger.info("Initialized " + str(layers_initialized) + \
                " conv2d layers using nn.init.kaiming_normal_")


def init_lin(model, logger=None):
    layers_initialized = 0
    a = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            layers_initialized += 1
            stdv = 1. / math.sqrt(m.weight.data.size(1))
            std_cur = stdv
            m.weight.data.fill_(std_cur)
            m.bias.data.fill_(std_cur)
    logger.info("Initialized " + str(layers_initialized) + " linear layers using PyTorch default")


def init_bn_01(model, logger=None):
    layers_initialized = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            layers_initialized += 1
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    logger.info("Initialized " + str(layers_initialized) + " BN layers using weight=1 and bias=0")


def init_bn_11(model, logger=None):
    layers_initialized = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            layers_initialized += 1
            m.weight.data.fill_(0.015)
            m.bias.data.fill_(0.015)
    logger.info("Initialized " + str(layers_initialized) + " BN layers using weight=0.015 and bias=0.015")


def init_bn_uniformweight(model, logger=None):
    layers_initialized = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            layers_initialized += 1
            m.weight.data.uniform_()
            m.bias.data.zero_()
    logger.info("Initialized " + str(layers_initialized) + " BN layers using weight=U(0,1) and bias=0")

###############


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def kaiming_normal_std_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where

    .. math::
        \text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan_in}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with 'relu' or 'leaky_relu' (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return std
