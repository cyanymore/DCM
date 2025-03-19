import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_lrelu=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))  # 下取整
        self.reflection_pad = nn.ReflectionPad2d(
            [reflection_padding, reflection_padding, reflection_padding, reflection_padding])
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.use_lrelu = use_lrelu
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_lrelu:
            out = self.lrelu(out)
            out = self.norm(out)
            # out = self.dropout(out)
        return out


# Dense Block unit
class res2net_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_size_neck, stride, width):
        super(res2net_Block, self).__init__()
        self.width = width
        convs = []
        self.conv1 = ConvLayer(in_channels, self.width * 4, kernel_size, stride, use_lrelu=True)
        for i in range(3):
            convs.append(ConvLayer(self.width, self.width, kernel_size_neck, stride, use_lrelu=True))
        self.convs = nn.ModuleList(convs)
        self.conv2 = ConvLayer(self.width * 4, out_channels, kernel_size, stride, use_lrelu=False)
        self.conv3 = ConvLayer(in_channels, out_channels, kernel_size, stride, use_lrelu=False)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        spx = torch.split(out1, self.width, 1)
        for i in range(3):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out2 = torch.cat((out, spx[3]), 1)
        out3 = self.conv2(out2)
        out4 = self.conv3(residual)
        OUT = self.lrelu(out3 + out4)
        return OUT


from torchsummary import summary

if __name__ == "__main__":
    input = torch.Tensor(1, 256, 256, 256).cuda()
    model = res2net_Block(256, 256, 1, 3, 1, 4).cuda()
    model.eval()
    print(model)
    output = model(input)
    summary(model, (256, 256, 256))
    print(output.shape)
