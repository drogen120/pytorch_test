import torch
import torch.nn as nn

def get_max_min(bit_width, f_width):
    i_width = bit_width - f_width
    base = torch.FloatTensor([2]).detach_()
    max_value = base.pow(i_width).detach_() - base.pow(-1*f_width).detach_()
    min_value = -1 * base.pow(i_width-1).detach_()
    return max_value.detach_().item(), min_value.detach_().item()

def fix(input_data, bit_width=16, f_width=14, is_training=True):
    max_value, min_value = get_max_min(bit_width, f_width)
    temp_data = input_data.clamp(min_value, max_value)
    temp_data.detach_()
    base = torch.FloatTensor([2]).detach_()
    pow_f32 = base.pow(-f_width).detach_()
    if is_training:
        temp_data.div_(pow_f32)
        random_numbers = torch.zeros_like(temp_data).uniform_(0.0,1.0)
        temp_data.add_(random_numbers).floor_()
    else:
        temp_data.div_(pow_f32).round_()
    return temp_data.clone()