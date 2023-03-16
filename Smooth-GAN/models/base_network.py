"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
from torch.nn import init
import torch


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def get_param_list(self, label):
        print("updating all params")
        return self.parameters()
        #raise NotImplementedError



class gen_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize, stride=1, rate=1, activation=nn.ELU()):
        """Define conv for generator

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            rate: Rate for or dilated conv.
            activation: Activation function after convolution.
        """
        p = int(rate*(ksize-1)/2)
        super(gen_conv, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
        self.activation = activation

    def forward(self, x):
        x = super(gen_conv, self).forward(x)
        if self.out_channels == 3 or self.activation is None:
            return x
        x, y = torch.split(x, int(self.out_channels/2), dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x

class gen_deconv(gen_conv):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv, self).__init__(cin, cout, ksize=3)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv, self).forward(x)
        return x