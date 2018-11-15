# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

class FixConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, fix_bit=4, bias=True):
        super(FixConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.fix_bit = fix_bit

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.weight.data.normal_()
        if bias is not None:
            self.bias.data.uniform_(-0.1,0.1)
    def get_max_min(self, bit_width, f_width):
        i_width = bit_width - f_width
        base = torch.FloatTensor([2]).detach_()
        max_value = base.pow(i_width).detach_() - base.pow(-1*f_width).detach_()
        min_value = -1 * base.pow(i_width-1).detach_()
        return max_value.detach_().item(), min_value.detach_().item()

    def fix_parameter(self, bit_width=16, f_width=8):
        max_value, min_value = self.get_max_min(bit_width, f_width)
        temp_weight = self.weight.data.clamp(min_value, max_value)
        temp_weight.detach_()
        if self.bias is not None:
            temp_bias = self.bias.data.clamp(min_value, max_value)
            temp_bias.detach_()
        base = torch.FloatTensor([2]).detach_()
        pow_f32 = base.pow(-f_width).detach_()
        temp_weight.div_(pow_f32).round_()
        temp_weight.mul_(pow_f32)
        if self.bias is not None:
            temp_bias.div_(pow_f32).round_()
            temp_bias.mul_(pow_f32)
            self.bias.data = temp_bias.clone()

        self.weight.data = temp_weight.clone()

    def forward(self, input):
        self.fix_parameter()
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding)

