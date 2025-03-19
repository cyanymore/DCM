from models.fusion_arc import *
from models.decoder_arc import *
from models.mamba_arc import *
from models.res2net_arc import *
import functools


class Mamba_branch(nn.Module):
    def __init__(self, ngf, norm_layer=nn.InstanceNorm2d):
        super(Mamba_branch, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.image_size = [256, 128, 64, 32]
        # add the outermost layer
        self.encoder_mb1 = MambaIR(img_size=self.image_size[0], embed_dim=ngf)

        self.down_mb1 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)

        # gradually reduce the number of filters from ngf * 8 to ngf
        self.encoder_mb2 = MambaIR(img_size=self.image_size[1], embed_dim=ngf * 2)

        self.down_mb2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.encoder_mb3 = MambaIR(img_size=self.image_size[2], embed_dim=ngf * 4)

        self.down_mb3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.encoder_mb4 = MambaIR(img_size=self.image_size[3], embed_dim=ngf * 8)

        self.down_mb4 = nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1, bias=use_bias)

    def forward(self, x):
        x1 = self.encoder_mb1(x)
        x2 = self.down_mb1(x1)
        x3 = self.encoder_mb2(x2)
        x4 = self.down_mb2(x3)
        x5 = self.encoder_mb3(x4)
        x6 = self.down_mb3(x5)
        x7 = self.encoder_mb4(x6)
        out = self.down_mb4(x7)

        return out, x7, x5, x3, x1


class Conv_branch(nn.Module):
    def __init__(self, ngf, norm_layer=nn.InstanceNorm2d):
        super(Conv_branch, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.encoder_cb1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            res2net_Block(in_channels=ngf, out_channels=ngf, kernel_size=1, kernel_size_neck=3, stride=1, width=4),
            norm_layer(ngf * 2),
        )

        self.down_cb1 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.encoder_cb2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            res2net_Block(in_channels=ngf * 2, out_channels=ngf * 2, kernel_size=1, kernel_size_neck=3, stride=1,
                          width=4),
            norm_layer(ngf * 4),
        )

        self.down_cb2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.encoder_cb3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            res2net_Block(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=1, kernel_size_neck=3, stride=1,
                          width=4),
            norm_layer(ngf * 8),
        )

        self.down_cb3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.encoder_cb4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            res2net_Block(in_channels=ngf * 8, out_channels=ngf * 8, kernel_size=1, kernel_size_neck=3, stride=1,
                          width=4),
            norm_layer(ngf * 16),
        )

        self.down_cb4 = nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1, bias=use_bias)

    def forward(self, x):
        x1 = self.encoder_cb1(x)
        x2 = self.down_cb1(x1)
        x3 = self.encoder_cb2(x2)
        x4 = self.down_cb2(x3)
        x5 = self.encoder_cb3(x4)
        x6 = self.down_cb3(x5)
        x7 = self.encoder_cb4(x6)
        out = self.down_cb4(x7)

        return out, x7, x5, x3, x1


class Decoder(nn.Module):
    def __init__(self, ngf):
        super(Decoder, self).__init__()

        self.decoder1 = DecoderBlock(ngf * 16, ngf * 8)
        self.decoder2 = DecoderBlock(ngf * 8, ngf * 4)
        self.decoder3 = DecoderBlock(ngf * 4, ngf * 2)
        self.decoder4 = DecoderBlock(ngf * 2, ngf)

    def forward(self, x, x_f1, x_f2, x_f3, x_f4):
        x1 = self.decoder1(x, x_f1)
        x2 = self.decoder2(x1, x_f2)
        x3 = self.decoder3(x2, x_f3)
        x4 = self.decoder4(x3, x_f4)

        return x4


class DCM(nn.Module):
    def __init__(self, in_ch, out_ch, ngf):
        super(DCM, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, ngf, kernel_size=3, stride=1, padding=1)

        self.encoder_mb = Mamba_branch(ngf=ngf)
        self.encoder_cb = Conv_branch(ngf=ngf)

        self.fmf = FMF(ngf * 16)

        self.decoder = Decoder(ngf)

        self.conv2 = nn.Conv2d(ngf, out_ch, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.conv1(x)
        x_mb, x_mb1, x_mb2, x_mb3, x_mb4 = self.encoder_mb(x1)
        x_cb, x_cb1, x_cb2, x_cb3, x_cb4 = self.encoder_cb(x1)
        x_fmf = self.fmf(x_mb, x_cb)
        x_f1 = x_mb1 + x_cb1
        x_f2 = x_mb2 + x_cb2
        x_f3 = x_mb3 + x_cb3
        x_f4 = x_mb4 + x_cb4
        x2 = self.decoder(x_fmf, x_f1, x_f2, x_f3, x_f4)
        x3 = x2 + x1
        out = self.tanh(self.conv2(x3))

        return out


from torchsummary import summary

if __name__ == "__main__":
    input = torch.Tensor(1, 3, 256, 256).cuda()
    model = DCM(3, 3, 32).cuda()
    model.eval()
    print(model)
    output = model(input)
    summary(model, (3, 256, 256))
    print(output[0].shape)
