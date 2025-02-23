import torch
import torch.nn as nn

class ConvAttention(nn.Module):


    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dwconv  = nn.Conv2d(in_dim, out_dim, 3, padding=1, groups = in_dim, bias =True)
        self.conv_11 = nn.Conv2d(out_dim, out_dim, (1,5), padding=(0,2),  bias =False)
        self.conv_12 = nn.Conv2d(out_dim, out_dim, (5,1), padding=(2,0),  bias =False)
        self.conv_21 = nn.Conv2d(out_dim, out_dim, (1,9), padding=(0,4),  bias =False)
        self.conv_22 = nn.Conv2d(out_dim, out_dim, (9,1), padding=(4,0),  bias =False)
        self.bn_ge1 = nn.Sequential(nn.BatchNorm2d(out_dim),
                                    nn.GELU())
        self.bn_ge2 = nn.Sequential(nn.BatchNorm2d(out_dim),
                                    nn.GELU())
        self.smooth  = nn.Conv2d(out_dim, out_dim, 1,  bias =False)
        self.spatial_wise = nn.Sequential(
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_dim, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        
        self.dwconv.apply(self.weights_init)
        self.conv_11.apply(self.weights_init)
        self.conv_12.apply(self.weights_init)
        self.conv_21.apply(self.weights_init)
        self.conv_22.apply(self.weights_init)
        self.bn_ge1.apply(self.weights_init)
        self.bn_ge2.apply(self.weights_init)
        self.spatial_wise.apply(self.weights_init)
        self.attention_wise.apply(self.weights_init)
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        
    def forward(self, x):
        y = self.dwconv(x)
        out1 = self.bn_ge1(self.conv_12(self.conv_11(y)))
        out2 = self.bn_ge2(self.conv_22(self.conv_21(y)))
        out3 = self.smooth(out1 + out2)
        out4 = torch.mean(out3, dim=1, keepdim=True)
        out5 = self.spatial_wise(out4) + out3
        out = self.attention_wise(out5) * x
        return out