import re
import types

import torch.nn
import torch.nn.init
from .SE_Attention import *
from .common import conv1x1_block, conv3x3_block, conv3x3_dw_block, conv5x5_dw_block, SEUnit, Classifier
class CGDF(torch.nn.Module):#defined for a CGDF block
    """
    CGDF used in CGDFNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,groups = 2):
        super().__init__()
        self.groups = groups
        self.conv_dw = conv3x3_dw_block(channels=in_channels, stride=stride)
        #self.conv_pw = conv1x1_block(in_channels=2*in_channels, out_channels=out_channels)
        self.depthwise_dilated = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=3,                                 
                                 stride=stride,
                                 padding=2,
                                 dilation=2,
                                 groups=in_channels,
                                 bias=True),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True)            
        )
        self.conv_pw_shuffle=torch.nn.Sequential(
                torch.nn.Conv2d(2*in_channels,
                                 out_channels,
                                 kernel_size=1,                                 
                                 stride=1,
                                 padding=0,
                                 groups=self.groups,
                                 bias=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)     
            )
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SE = SE(out_channels, 16)
    def forward(self, x):
        residual = x
        
        x_dilated1 = self.conv_dw(x)        
        x_dilated2 = self.depthwise_dilated(x)
        
        x = torch.cat((x_dilated1,x_dilated2),1)
        x = self.channel_shuffle(x)
        #x = self.conv_pw(x)
        x = self.conv_pw_shuffle(x)
        x = self.SE(x)
        if self.stride == 1 and self.in_channels == self.out_channels:                    
            x = x + residual
        return x
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.groups == 0
        group_channels = num_channels // self.groups
        
        x = x.reshape(batchsize, group_channels, self.groups, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)
      
        return x
    def channel_shuffle1(self,x):
        """channel shuffle operation
        Args:
            x: input tensor
            groups: input branch number
        """
    
        batch_size, channels, height, width = x.size()
        channels_per_group = int(channels // self.group)
    
        x = x.view(batch_size, self.group, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
    
        return x

class CGDFNet(torch.nn.Module):
    """
    Class for constructing MobileNetsV1.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True,groups=2):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                stage.add_module("unit{}".format(unit_id + 1), CGDF(in_channels=in_channels, out_channels=unit_channels, stride=stride,groups=groups))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        
        self.final_conv_channels = 1024        
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        self.backbone.add_module("dropout1",torch.nn.Dropout2d(0.2))
        
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        
        self.backbone.add_module("dropout2",torch.nn.Dropout2d(0.2))
        in_channels = self.final_conv_channels
        
        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():            
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)            
            
            elif isinstance(module, torch.nn.Linear):                
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):                
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            
        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def build_CGDFNet(num_classes, width_multiplier=1.0, cifar=False,groups=2):
    """
    
    """

    init_conv_channels = 32    
    channels = [[64], [128, 128], [256, 256], [512, 512, 512, 512], [512]]

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    if width_multiplier != 1.0:
        channels = [[int(unit * width_multiplier) for unit in stage] for stage in channels]
        init_conv_channels = int(init_conv_channels * width_multiplier)

    return CGDFNet(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size,groups=groups)
