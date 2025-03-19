import torch
import torch.nn as nn
import torch.nn.functional as F


class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #

        return x


class AlignedModulev2(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModulev2, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 4, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, low, high):
        low_feature, h_feature = low, high
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)
        fuse_feature = h_feature_warp + l_feature_warp

        return fuse_feature, l_feature_warp, h_feature_warp

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class DecoderBlock(nn.Module):  # Cross-scale Multi-level Feature Fusion Module
    def __init__(self, in_chan, out_chan, reduction_ratio=16):
        super(DecoderBlock, self).__init__()
        self.ca1 = Channel_Att(in_chan)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.fa1 = AlignedModulev2(out_chan, out_chan)
        self.ca2 = Channel_Att(out_chan)
        self.sa = nn.Sequential(
            nn.Conv2d(out_chan, out_chan // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_chan // reduction_ratio),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_chan // reduction_ratio, out_chan // reduction_ratio, kernel_size=3, padding=4, dilation=4),
            nn.InstanceNorm2d(out_chan // reduction_ratio),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_chan // 16, out_chan, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, high, low):
        high = self.ca1(high)
        high = self.conv1(high)
        fuse, low, high = self.fa1(low, high)
        # print(fuse.shape,high.shape,low.shape)
        f_c = self.ca2(fuse)
        f_s = self.sa(fuse)
        weight = torch.sigmoid(f_s + f_c)
        x = high * weight + (1 - weight) * low
        return x


from torchsummary import summary

if __name__ == "__main__":
    input = torch.Tensor(1, 256, 32, 32).cuda()
    input1 = torch.Tensor(1, 128, 64, 64).cuda()
    model = DecoderBlock(256, 128).cuda()
    model.eval()
    print(model)
    output = model(input, input1)
    summary(model, [(256, 32, 32), (128, 64, 64)])
    print(output.shape)
