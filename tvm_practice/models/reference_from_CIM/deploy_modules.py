import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safe_convert import *
from quantize import min_max_quantize


def make_layer_state_dict(state_dict, layer_index):
  downsample_key = f"module.layer{layer_index}.0.downsample.0.weight"
  s_dict = state_dict["state_dict"]
  downsample_weight = s_dict[downsample_key].cpu(
  ) if downsample_key in s_dict else None

  downsample_weight_s_key = f"module.layer{layer_index}.0.downsample.0.quant_func.s"
  downsample_act_s_key = f"module.layer{layer_index}.0.downsample.0.act_func.s"
  downsample_act_s = s_dict[downsample_act_s_key].cpu(
  ) if downsample_act_s_key in s_dict else None
  downsample_weight_s = s_dict[downsample_weight_s_key].cpu(
  ) if downsample_weight_s_key in s_dict else None

  bn1_var = s_dict[f"module.layer{layer_index}.0.bn1.running_var"].cpu()
  bn1_gamma = s_dict[f"module.layer{layer_index}.0.bn1.weight"].cpu()
  bn1_mean = s_dict[f"module.layer{layer_index}.0.bn1.running_mean"].cpu()
  bn1_beta = s_dict[f"module.layer{layer_index}.0.bn1.bias"].cpu()
  bn2_var = s_dict[f"module.layer{layer_index}.0.bn2.running_var"].cpu()
  bn2_gamma = s_dict[f"module.layer{layer_index}.0.bn2.weight"].cpu()
  bn2_mean = s_dict[f"module.layer{layer_index}.0.bn2.running_mean"].cpu()
  bn2_beta = s_dict[f"module.layer{layer_index}.0.bn2.bias"].cpu()

  temp_dict = {
      'act1_s': s_dict[f"module.layer{layer_index}.0.act1.s"].cpu(),
      'conv1_weight': s_dict[f"module.layer{layer_index}.0.conv1.weight"].cpu(),
      'conv1_weight_s': s_dict[f"module.layer{layer_index}.0.conv1.quant_func.s"].cpu(),
      'bn1_scale': bn1_gamma / torch.sqrt(bn1_var),
      'bn1_bias': bn1_beta - bn1_gamma * bn1_mean / torch.sqrt(bn1_var),
      'act2_s': s_dict[f"module.layer{layer_index}.0.act2.s"].cpu(),
      'conv2_weight': s_dict[f"module.layer{layer_index}.0.conv2.weight"].cpu(),
      'conv2_weight_s': s_dict[f"module.layer{layer_index}.0.conv2.quant_func.s"].cpu(),
      'bn2_scale': bn2_gamma / torch.sqrt(bn2_var),
      'bn2_bias': bn2_beta - bn2_gamma * bn2_mean / torch.sqrt(bn2_var),
      'downsample_weight': downsample_weight,
      'downsample_weight_s': downsample_weight_s,
      'downsample_act_s': downsample_act_s,
  }
  return temp_dict


def make_conv1_bn1_fc_state_dict(state_dict):
  """Extract conv1, bn1, and fc state dict from the model state."""
  s_dict = state_dict["state_dict"]
  temp_dict = {
      'conv1_weight': s_dict["module.conv1.weight"].cpu(),
      'conv1_bias': s_dict["module.conv1.bias"].cpu(),
      'bn1_weight': s_dict["module.bn1.weight"].cpu(),
      'bn1_bias': s_dict["module.bn1.bias"].cpu(),
      'bn1_running_mean': s_dict["module.bn1.running_mean"].cpu(),
      'bn1_running_var': s_dict["module.bn1.running_var"].cpu(),
      'fc_weight': s_dict["module.fc.weight"].cpu(),
      'fc_bias': s_dict["module.fc.bias"].cpu(),
  }
  return temp_dict


class Q_act(nn.Module):
  def __init__(self, scale, dequantize=False):
    super().__init__()
    self.register_buffer('scale', scale)
    self.dequantize = dequantize

  def forward(self, x):
    out = torch.clamp(torch.round(x / self.scale), 0, 15)
    if self.dequantize:
      out = out * self.scale
    return out

  def __repr__(self):
    return f'Q_act(scale={self.scale:.3f})'


class MMQuant(nn.Module):
  def __init__(self, min, max):
    super().__init__()
    self.register_buffer('min', min)
    self.register_buffer('max', max)
    thresholds = []
    for i in range(15):
      offset = (i + 1) * (int(max) - int(min))
      norm_offset = np.floor(offset / 16.0)
      thresholds.append(min + norm_offset)
    self.register_buffer('thresholds', torch.tensor(thresholds))

  def forward(self, x):
    out = min_max_quantize(to_int16(x), self.min, self.max)
    return out

  def __repr__(self):
    return f'MMQuant(min={self.min}, max={self.max})'


class Conv(nn.Module):
  def __init__(self, weight, stride=1, padding=1):
    super().__init__()
    self.register_buffer('weight', weight)
    self.stride = stride
    self.padding = padding

  def forward(self, x):
    out_dtype = x.dtype

    # explicit float conversion to prevent CUDNN error
    out = F.conv2d(x.float(), self.weight.float(), bias=None,
                   stride=self.stride, padding=self.padding)

    # convert back to original dtype (prevent CUDNN error but retain dtype)
    out = out.to(out_dtype)
    return out

  def __repr__(self):
    return f'Conv(weight_shape={list(self.weight.shape)}, stride={self.stride}, padding={self.padding})'


class MultiplyAdd(nn.Module):
  def __init__(self, scale, bias, use_int16=False):
    super().__init__()
    self.register_buffer('scale', scale)
    self.register_buffer('bias', bias)
    self.use_int16 = use_int16

  def forward(self, x):
    out = x * self.scale[:, None, None] + self.bias[:, None, None]

    if self.use_int16:
      out = to_int16(out)
    return out

  def __repr__(self):
    return f'MultiplyAdd(num_features={len(self.scale)}, avg_scale={self.scale.to(torch.float32).mean():.3f}, avg_bias={self.bias.to(torch.float32).mean():.3f})'


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, stride=1, **kwargs):
    super().__init__()
    self._setup(stride=stride, **kwargs)

  def _setup(self, stride=1, **kwargs):
    self.act1 = Q_act(kwargs['act1_s'], dequantize=True)
    self.conv1 = Conv(kwargs['conv1_weight'], stride=stride, padding=1)
    self.bn1 = MultiplyAdd(kwargs['bn1_scale'], kwargs['bn1_bias'])
    self.act2 = Q_act(kwargs['act2_s'], dequantize=True)
    self.conv2 = Conv(kwargs['conv2_weight'], stride=1, padding=1)
    self.bn2 = MultiplyAdd(kwargs['bn2_scale'], kwargs['bn2_bias'])

    if kwargs['downsample_weight'] is not None:
      # 1x1 conv has no padding
      self.downsample = nn.Sequential(
          Q_act(kwargs['downsample_act_s'], dequantize=True),
          Conv(kwargs['downsample_weight'], stride=stride, padding=0))
    else:
      self.downsample = None

  def forward(self, x):
    identity = x

    out = self.act1(x)
    out = self.conv1(out)
    out = self.bn1(out)
    out = self.act2(out)
    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity

    return out


class BasicBlockQConv(BasicBlock):
  def _setup(self, stride=1, **kwargs):
    conv1_weight_q = torch.round(torch.clamp(
        kwargs['conv1_weight'] / kwargs['conv1_weight_s'], -8, 7))
    conv2_weight_q = torch.round(torch.clamp(
        kwargs['conv2_weight'] / kwargs['conv2_weight_s'], -8, 7))
    bn1_scale = kwargs['act1_s'] * \
        kwargs['conv1_weight_s'] * kwargs['bn1_scale']
    bn1_bias = kwargs['bn1_bias']
    bn2_scale = kwargs['act2_s'] * \
        kwargs['conv2_weight_s'] * kwargs['bn2_scale']
    bn2_bias = kwargs['bn2_bias']

    self.act1 = Q_act(kwargs['act1_s'], dequantize=False)
    self.conv1 = Conv(conv1_weight_q, stride=stride, padding=1)
    self.bn1 = MultiplyAdd(bn1_scale, bn1_bias)
    self.act2 = Q_act(kwargs['act2_s'], dequantize=False)
    self.conv2 = Conv(conv2_weight_q, stride=1, padding=1)
    self.bn2 = MultiplyAdd(bn2_scale, bn2_bias)

    if kwargs['downsample_weight'] is not None:
      downsample_weight_q = torch.round(torch.clamp(
          kwargs['downsample_weight'] / kwargs['downsample_weight_s'], -8, 7))
      downsample_scale = torch.ones_like(bn1_scale) * kwargs['downsample_weight_s'] * kwargs['downsample_act_s']
      downsample_bias = torch.zeros_like(bn1_scale)
      # 1x1 conv has no padding
      self.downsample = nn.Sequential(
          Q_act(kwargs['downsample_act_s'], dequantize=False),
          Conv(downsample_weight_q, stride=stride, padding=0),
          MultiplyAdd(downsample_scale, downsample_bias))  # TODO: quantize this
    else:
      self.downsample = None


class BasicBlockIMCFlow(nn.Module):
  expansion = 1

  def __init__(self, stride=1, is_fp_input=True, is_fp_output=True, **kwargs):
    super().__init__()
    self.block_int16 = BasicBlockIMCFlowInt16(stride=stride, **kwargs)
    self.is_fp_input = is_fp_input
    self.is_fp_output = is_fp_output

  def forward(self, x):
    # input quant (CPU)
    if self.is_fp_input:
      x_int16 = to_int16(x * self.block_int16.x_f)
    else:
      x_int16 = to_int16(x)

    out = self.block_int16(x_int16)

    # output dequant (CPU)
    if self.is_fp_output:
      out = out / self.block_int16.bn2_f
    return out


class BasicBlockIMCFlowInt16(nn.Module):
  expansion = 1

  def __init__(self, stride=1, **kwargs):
    super().__init__()
    self._setup(stride=stride, **kwargs)

  def _setup(self, stride=1, **kwargs):
    conv1_weight_q = to_int4(torch.round(torch.clamp(
        kwargs['conv1_weight'] / kwargs['conv1_weight_s'], -8, 7)))
    conv2_weight_q = to_int4(torch.round(torch.clamp(
        kwargs['conv2_weight'] / kwargs['conv2_weight_s'], -8, 7)))

    act1_min = to_int16(-0.5 * kwargs['act1_s'] * kwargs['x_f'], assert_range=True)
    act1_max = to_int16(15.5 * kwargs['act1_s'] * kwargs['x_f'], assert_range=True)
    act2_min = to_int16(-0.5 * kwargs['act2_s'] * kwargs['bn1_f'], assert_range=True)
    act2_max = to_int16(15.5 * kwargs['act2_s'] * kwargs['bn1_f'], assert_range=True)

    bn1_scale = to_int16(
        kwargs['act1_s'] * kwargs['conv1_weight_s'] * kwargs['bn1_scale'] * kwargs['bn1_f'], assert_range=True)
    bn1_bias = to_int16(kwargs['bn1_bias'] * kwargs['bn1_f'], assert_range=True)
    bn2_scale = to_int16(
        kwargs['act2_s'] * kwargs['conv2_weight_s'] * kwargs['bn2_scale'] * kwargs['bn2_f'], assert_range=True)
    bn2_bias = to_int16(kwargs['bn2_bias'] * kwargs['bn2_f'], assert_range=True)

    self.act1 = MMQuant(act1_min, act1_max)
    self.conv1 = Conv(conv1_weight_q, stride=stride, padding=1)
    self.bn1 = MultiplyAdd(bn1_scale, bn1_bias)
    self.act2 = MMQuant(act2_min, act2_max)
    self.conv2 = Conv(conv2_weight_q, stride=1, padding=1)
    self.bn2 = MultiplyAdd(bn2_scale, bn2_bias)

    # save scale factors for dequantization
    self.register_buffer('x_f', torch.tensor(kwargs['x_f']))
    self.register_buffer('conv1_weight_s', kwargs['conv1_weight_s'])
    self.register_buffer('conv2_weight_s', kwargs['conv2_weight_s'])
    self.register_buffer('act1_s', kwargs['act1_s'])
    self.register_buffer('act2_s', kwargs['act2_s'])
    self.register_buffer('bn1_f', torch.tensor(kwargs['bn1_f']))
    self.register_buffer('bn2_f', torch.tensor(kwargs['bn2_f']))

    # Create downsample if downsample_weight exists
    if kwargs['downsample_weight'] is not None:
      downsample_weight_q = to_int4(torch.round(torch.clamp(
          kwargs['downsample_weight'] / kwargs['downsample_weight_s'], -8, 7)))
      # 1x1 conv has no padding
      ds_act_min = to_int16(-0.5 * kwargs['downsample_act_s'] * kwargs['x_f'], assert_range=True)
      ds_act_max = to_int16(15.5 * kwargs['downsample_act_s'] * kwargs['x_f'], assert_range=True)
      downsample_scale = torch.ones_like(
        bn1_scale) * to_int16(kwargs['downsample_weight_s'] * kwargs['downsample_act_s'] * kwargs['bn2_f'])
      downsample_bias = torch.zeros_like(bn1_scale)
      self.downsample = nn.Sequential(
          MMQuant(ds_act_min, ds_act_max),
          Conv(downsample_weight_q, stride=stride, padding=0),
          MultiplyAdd(downsample_scale, downsample_bias))  # TODO: quantize this
    else:
      self.downsample = None

  def forward(self, x):
    assert in_range(x, -2**15, 2**15 - 1), "Input is out of range for int16"
    assert x.dtype == torch.int32, "Input must be of type int32"

    out = self.act1(x)
    out = self.conv1(out)
    out = self.bn1(out)
    out = self.act2(out)
    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is None:
      identity = x * to_int16(self.bn2_f / self.x_f, assert_range=True)
    else:
      identity = to_int16(self.downsample(x), assert_range=False)

    out += identity

    return to_int16(out, assert_range=False)


class ResNet8(nn.Module):
  def __init__(self, state_dict):
    super().__init__()

    layer_state_dicts = {
        "layer1": make_layer_state_dict(state_dict, 1),
        "layer2": make_layer_state_dict(state_dict, 2),
        "layer3": make_layer_state_dict(state_dict, 3),
    }
    conv1_bn1_fc_state = make_conv1_bn1_fc_state_dict(state_dict)

    self.inplanes = 16

    self.conv1 = nn.Conv2d(
        3, self.inplanes, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(self.inplanes)

    self.layer1 = BasicBlockQConv(stride=1, **layer_state_dicts["layer1"])
    self.layer2 = BasicBlockQConv(stride=2, **layer_state_dicts["layer2"])
    self.layer3 = BasicBlockQConv(stride=2, **layer_state_dicts["layer3"])

    self.relu = nn.ReLU(inplace=True)
    # # this layer works for any size of input.
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(64, 10)  # assume that Last layer is FP

    # Load state dicts for conv1, bn1, and fc
    self.conv1.weight.data = conv1_bn1_fc_state['conv1_weight']
    self.conv1.bias.data = conv1_bn1_fc_state['conv1_bias']
    self.bn1.weight.data = conv1_bn1_fc_state['bn1_weight']
    self.bn1.bias.data = conv1_bn1_fc_state['bn1_bias']
    self.bn1.running_mean.data = conv1_bn1_fc_state['bn1_running_mean']
    self.bn1.running_var.data = conv1_bn1_fc_state['bn1_running_var']
    self.fc.weight.data = conv1_bn1_fc_state['fc_weight']
    self.fc.bias.data = conv1_bn1_fc_state['fc_bias']

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.relu(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


class ResNet8IMCFlow(nn.Module):
  def __init__(self, state_dict, adjust_factors):
    super().__init__()

    self.adjust_factors = adjust_factors

    layer_state_dicts = {
        "layer1": make_layer_state_dict(state_dict, 1),
        "layer2": make_layer_state_dict(state_dict, 2),
        "layer3": make_layer_state_dict(state_dict, 3),
    }
    conv1_bn1_fc_state = make_conv1_bn1_fc_state_dict(state_dict)

    self.inplanes = 16

    self.conv1 = nn.Conv2d(
        3, self.inplanes, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(self.inplanes)

    self.layer1 = BasicBlockIMCFlow(
        stride=1, is_fp_input=True, is_fp_output=False,
        x_f=self.adjust_factors['x_f_1'], bn1_f=self.adjust_factors['bn1_f_1'], bn2_f=self.adjust_factors['bn2_f_1'],
        **layer_state_dicts["layer1"])
    self.layer2 = BasicBlockIMCFlow(
        stride=2, is_fp_input=False, is_fp_output=False,
        x_f=self.adjust_factors['x_f_2'], bn1_f=self.adjust_factors['bn1_f_2'], bn2_f=self.adjust_factors['bn2_f_2'],
        **layer_state_dicts["layer2"])
    self.layer3 = BasicBlockIMCFlow(
        stride=2, is_fp_input=False, is_fp_output=True,
        x_f=self.adjust_factors['x_f_3'], bn1_f=self.adjust_factors['bn1_f_3'], bn2_f=self.adjust_factors['bn2_f_3'],
        **layer_state_dicts["layer3"])

    self.relu = nn.ReLU(inplace=True)
    # this layer works for any size of input.
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(64, 10)  # assume that Last layer is FP

    # Load state dicts for conv1, bn1, and fc
    self.conv1.weight.data = conv1_bn1_fc_state['conv1_weight']
    self.conv1.bias.data = conv1_bn1_fc_state['conv1_bias']
    self.bn1.weight.data = conv1_bn1_fc_state['bn1_weight']
    self.bn1.bias.data = conv1_bn1_fc_state['bn1_bias']
    self.bn1.running_mean.data = conv1_bn1_fc_state['bn1_running_mean']
    self.bn1.running_var.data = conv1_bn1_fc_state['bn1_running_var']
    self.fc.weight.data = conv1_bn1_fc_state['fc_weight']
    self.fc.bias.data = conv1_bn1_fc_state['fc_bias']

  def forward(self, x):

    x = self.conv1(x)
    x = self.bn1(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.relu(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x
