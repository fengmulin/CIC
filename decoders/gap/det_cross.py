from collections import OrderedDict

import torch
import torch.nn as nn
from ..tools.ohem import ohem_batch
from ..tools.attention import ConvAttention
BatchNorm2d = nn.BatchNorm2d

class Det_cross(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10, alpha=16,beta=9,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(Det_cross, self).__init__()
        #self.k = nn.Parameter(torch.ones(1)*7)
        self.serial = serial
        self.layernum = 4
        self.qzz2= nn.ReLU(inplace=True)
        self.qzz =  nn.Sigmoid()
        self.beta = nn.Parameter(torch.ones(1)*beta)
        self.alpha = nn.Parameter(torch.ones(1)*alpha)
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      self.layernum, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//self.layernum, 3, padding=1, bias=bias)

        self.binarize1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      self.layernum, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//self.layernum),
            nn.ReLU(inplace=True),)
        self.binarize2 = nn.Sequential(
            nn.Conv2d(inner_channels//2, inner_channels//self.layernum, 1, bias=bias),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//self.layernum, 2, 2),
            BatchNorm2d(inner_channels//self.layernum),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//self.layernum, 1, 2, 2),
            nn.Sigmoid())
        
        self.thresh1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      self.layernum, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//self.layernum),
            nn.ReLU(inplace=True),)
        
        self.thresh2 = nn.Sequential(
            nn.ConvTranspose2d(inner_channels//self.layernum, inner_channels//self.layernum, 2, 2),
            BatchNorm2d(inner_channels//self.layernum),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//self.layernum, 1, 2, 2),
            nn.Sigmoid())
        
        self.mix = nn.Sequential(
            nn.Conv2d(inner_channels//2, inner_channels //
                      2, 1, padding=0, bias=bias),
            BatchNorm2d(inner_channels//2),
            nn.ReLU(inplace=True))
        
        self.mix_attention = ConvAttention(inner_channels // 2, inner_channels // 2)
        
        self.binarize1.apply(self.weights_init)
        self.thresh1.apply(self.weights_init)
        self.binarize2.apply(self.weights_init)
        self.thresh2.apply(self.weights_init)
        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
            
    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        fuse = torch.cat((p5, p4, p3, p2), 1)

        kernel_fuse = self.binarize1(fuse)
        text_fuse = self.thresh1(fuse)
        fuse_mix = torch.cat((kernel_fuse, text_fuse), 1)
        fuse_mix = self.mix(fuse_mix)
        fuse_mix = self.mix_attention(fuse_mix)
        binary = self.binarize2(fuse_mix)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
       
        if self.training:

            ori_binary = self.thresh2(text_fuse)      
            gap = self.qzz(self.alpha*ori_binary - self.alpha*binary - self.beta)
            result.update(ori_binary=ori_binary, gap=gap)
        return result
