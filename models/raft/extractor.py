import torch
import torch.nn as nn
import torch.nn.functional as F
        

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, pad_mode='zeros'):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode=pad_mode)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode=pad_mode)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding_mode=pad_mode), self.norm3)
            

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, pad_mode='zeros'):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0, padding_mode=pad_mode)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride, padding_mode=pad_mode)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0, padding_mode=pad_mode)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding_mode=pad_mode), self.norm4)
            

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder(nn.Module):
    def __init__(self, layers_dims=[64, 64, 96, 128], output_dim=128, norm_fn='batch', dropout=0.0, pad_mode='zeros'):
        super(BasicEncoder, self).__init__()
        self.in_planes = layers_dims[0]
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(self.in_planes)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode=pad_mode)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(layers_dims[1], stride=1, pad_mode=pad_mode)
        self.layer2 = self._make_layer(layers_dims[2], stride=2, pad_mode=pad_mode)
        self.layer3 = self._make_layer(layers_dims[3], stride=2, pad_mode=pad_mode)

        # output convolution
        self.conv2 = nn.Conv2d(layers_dims[3], output_dim, kernel_size=1, padding_mode=pad_mode)
        
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, pad_mode='zeros'):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride, pad_mode=pad_mode)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1, pad_mode=pad_mode)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        xf = self.conv2(x3)

        if self.training and self.dropout is not None:
            xf = self.dropout(xf)

        return xf


class SmallEncoder(nn.Module):
    def __init__(self, layers_dims=[32, 32, 64, 96], output_dim=128, norm_fn='batch', dropout=0.0, pad_mode='zeros'):
        super(SmallEncoder, self).__init__()
        self.in_planes = layers_dims[0]
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(self.in_planes)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode=pad_mode)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(layers_dims[1], stride=1, pad_mode=pad_mode)
        self.layer2 = self._make_layer(layers_dims[2], stride=2, pad_mode=pad_mode)
        self.layer3 = self._make_layer(layers_dims[3], stride=2, pad_mode=pad_mode)
        
        self.conv2 = nn.Conv2d(layers_dims[3], output_dim, kernel_size=1, padding_mode=pad_mode)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, pad_mode='zeros'):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride, pad_mode=pad_mode)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1, pad_mode=pad_mode)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        xf = self.conv2(x3)

        if self.training and self.dropout is not None:
            xf = self.dropout(xf)

        return xf
