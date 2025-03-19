import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class DWConv(nn.Module):
    def __init__(self, dim=768, ues_se=True):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.se = None
        if ues_se:
            self.se = SEBlock(channels=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        if self.se:
            x = self.se(self.dwconv(x))
        else:
            x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out


class vit_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        """
        此函数用于初始化相关参数
        :param dim: 输入token的维度
        :param num_heads: 注意力多头数量
        :param qkv_bias: 是否使用偏置，默认False
        :param qk_scale: 缩放因子
        :param attn_drop_ratio: 注意力的比例
        :param proj_drop_ratio: 投影的比例
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 计算每一个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 得到根号d_k分之一的值
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 通过全连接层生成得到qkv
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        """
        此函数用于前向传播
        :param x: 输入序列
        :return: 处理后的序列
        """
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class vit_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class vit_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = vit_Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.attn = vit_Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = vit_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FMF(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)

        self.fc1_2 = nn.Linear(channels, channels // reduction)
        self.fc2_2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

        self.swin1 = vit_Block(dim=channels, num_heads=8)
        self.swin2 = vit_Block(dim=channels, num_heads=8)

        self.dwconv = DWConv(2 * channels, ues_se=False)
        self.conv1x1 = conv1x1(2 * channels, channels)
        self.se = SEBlock(channels)

    def forward(self, x1, x2):
        b, c, h, w = x1.size()
        out1 = self.avg_pool(x1).view(b, c)
        out2 = self.avg_pool(x2).view(b, c)

        out1 = self.fc2(self.relu(self.fc1(out1)))
        out2 = self.fc2_2(self.relu(self.fc1_2(out2)))

        out1, out2 = self.sigmoid(out1).view(b, c, 1), self.sigmoid(out2).view(b, c, 1)
        out1 = out1.flatten(2).transpose(1, 2)  # B,1,C
        out2 = out2.flatten(2).transpose(1, 2)  # B,1,C
        x1 = x1.flatten(2).transpose(1, 2)  # B,L,C
        x2 = x2.flatten(2).transpose(1, 2)  # B,L,C

        out2 = torch.cat((x1, out2), dim=1)
        out1 = torch.cat((x2, out1), dim=1)

        out2 = self.swin1(out2)[:, :h * w, :]
        out1 = self.swin2(out1)[:, :h * w, :]

        out1 = out1.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        out2 = out2.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        out = self.dwconv(torch.cat((out1, out2), dim=1).flatten(2).transpose(1, 2), h, w).reshape(b, h, w, -1).permute(
            0, 3, 1, 2).contiguous()
        out = self.se(self.conv1x1(out))
        return out


from torchsummary import summary

if __name__ == "__main__":
    input = torch.Tensor(1, 256, 32, 32).cuda()
    input1 = torch.Tensor(1, 256, 32, 32).cuda()
    model = FMF(256).cuda()
    model.eval()
    print(model)
    output = model(input, input1)
    summary(model, [(256, 32, 32), (256, 32, 32)])
    print(output.shape)
