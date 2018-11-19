# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

class FixConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, fix_bit=4, bias=True, training=True):
        super(FixConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.fix_bit = fix_bit
        self.training = training

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.weight.data.normal_()
        if bias is not None:
            self.bias.data.uniform_(-0.1,0.1)

    def fix_parameter(self, bit_width=16, f_width=14):
        self.weight.data = utils.fix(self.weight.data, bit_width=bit_width, 
            f_width=f_width, is_training=self.training)
        if self.bias is not None:
            self.bias.data = utils.fix(self.bias.data, bit_width=bit_width,
            f_width=f_width, is_training=self.training)
        # max_value, min_value = self.get_max_min(bit_width, f_width)
        # temp_weight = self.weight.data.clamp(min_value, max_value)
        # temp_weight.detach_()
        # if self.bias is not None:
        #     temp_bias = self.bias.data.clamp(min_value, max_value)
        #     temp_bias.detach_()
        # base = torch.FloatTensor([2]).detach_()
        # pow_f32 = base.pow(-f_width).detach_()
        # if self.training:
        #     temp_weight.div_(pow_f32)
        #     random_numbers = torch.zeros_like(temp_weight).uniform_(-0.5,0.5)
        #     temp_weight.add_(random_numbers).round_()
        # else:
        #     temp_weight.div_(pow_f32).round_()
        # temp_weight.mul_(pow_f32)
        # if self.bias is not None:
        #     if self.training:
        #         temp_bias.div_(pow_f32)
        #         random_numbers = torch.zeros_like(temp_bias).uniform_(-0.5,0.5)
        #         temp_bias.add_(random_numbers).round_()               
        #     else:
        #         temp_bias.div_(pow_f32).round_()

        #     temp_bias.mul_(pow_f32)
        #     self.bias.data = temp_bias.clone()

        # self.weight.data = temp_weight.clone()

    def forward(self, input):
        self.fix_parameter()
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding)

class FixLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, training=True): 
        super(FixLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1,0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1,0.1)

    def fix_parameter(self, bit_width=16, f_width=8):
        self.weight.data = utils.fix(self.weight.data, bit_width=bit_width, 
            f_width=f_width, is_training=self.training)
        if self.bias is not None:
            self.bias.data = utils.fix(self.bias.data, bit_width=bit_width,
            f_width=f_width, is_training=self.training)

    def forward(self, input_data):
        self.fix_parameter()
        return F.LinearFunction.apply(input_data, self.weight, self.bias)