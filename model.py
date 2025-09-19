import torch

import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
# from .lsknet import lsknet_b0, lsknet_b1
import math
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class Mlp(nn.Module):
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


class ATMOp(nn.Module):
    def __init__(
            self, in_chans, out_chans, stride: int = 1, padding: int = 0, dilation: int = 1,
            bias: bool = True, dimension: str = ''
    ):
        super(ATMOp, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.dimension = dimension

        self.weight = nn.Parameter(torch.empty(out_chans, in_chans, 1, 1))  # kernel_size = (1, 1)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_chans))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        """
        ATM along one dimension, the shape will not be changed
        input: [B, C, H, W]
        offset: [B, C, H, W]
        """
        B, C, H, W = input.size()
        offset_t = torch.zeros(B, 2 * C, H, W, dtype=input.dtype, layout=input.layout, device=input.device)
        if self.dimension == 'w':
            offset_t[:, 1::2, :, :] += offset
        elif self.dimension == 'h':
            offset_t[:, 0::2, :, :] += offset
        else:
            raise NotImplementedError(f"{self.dimension} dimension not implemented")
        return deform_conv2d(
            input, offset_t, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation
        )

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'dimension={dimension}'
        s += ', in_chans={in_chans}'
        s += ', out_chans={out_chans}'
        s += ', stride={stride}'
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class ATMBlock(nn.Module):
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.dim = dim

        self.atm_c = nn.Linear(dim, dim, bias=False)
        self.atm_h = ATMOp(dim, dim, dimension='h')
        self.atm_w = ATMOp(dim, dim, dimension='w')

        self.fusion = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, offset):
        """
        x: [B, H, W, C]
        offsets: [B, 2C, H, W]
        """
        B, H, W, C = x.shape
        assert offset.shape == (B, 2 * C, H, W), f"offset shape not match, got {offset.shape}"
        w = self.atm_w(x.permute(0, 3, 1, 2), offset[:, :C, :, :]).permute(0, 2, 3, 1)
        h = self.atm_h(x.permute(0, 3, 1, 2), offset[:, C:, :, :]).permute(0, 2, 3, 1)
        c = self.atm_c(x)

        a = (w + h + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.fusion(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = w * a[0] + h * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + ' ('
        s += 'dim: {dim}'
        s += ')'
        return s.format(**self.__dict__)


class ActiveBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 share_dim=1, new_offset=False,
                 ):
        super().__init__()
        factor = 2
        d_model = int(dim // factor)
        self.down = nn.Linear(dim, d_model)
        self.up = nn.Linear(d_model, dim)
        self.norm1 = norm_layer(d_model)
        self.atm = ATMBlock(d_model)
        self.norm2 = norm_layer(d_model)
        self.mlp = Mlp(in_features=d_model, hidden_features=int(d_model * mlp_ratio), act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.new_offset = new_offset
        self.share_dim = share_dim

        if new_offset:
            self.offset_layer = nn.Sequential(
                norm_layer(d_model),
                nn.Linear(d_model, d_model * 2 // self.share_dim)
            )
        else:
            self.offset_layer = None

    def forward(self, x, offset=None):
        """
        :param x: [B, H, W, C]
        :param offset: [B, 2C, H, W]
        """
        input_x = self.down(x)
        if self.offset_layer and offset is None:
            offset = self.offset_layer(input_x).repeat_interleave(self.share_dim, dim=-1).permute(0, 3, 1, 2)

        input_x = input_x + self.drop_path(self.atm(self.norm1(input_x), offset))
        x = self.up(input_x) + x

        if self.offset_layer:
            return x, offset
        else:
            return x

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + ' ('
        s += 'new_offset: {offset}'
        s += ', share_dim: {share_dim}'
        s += ')'
        return s.format(**self.__dict__)


class PEG(nn.Module):

    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PEG, self).__init__()
        # depth conv
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=stride, padding=1, bias=True, groups=embed_dim)
        self.stride = stride

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x_conv = x
        x_conv = x_conv.permute(0, 3, 1, 2)
        if self.stride == 1:
            x = self.proj(x_conv) + x_conv
        else:
            x = self.proj(x_conv)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale

        if dim_scale == 2:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
            self.adjust_channels = nn.Conv2d(dim // 4, dim // dim_scale, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.pixel_shuffle = nn.Identity()
            self.adjust_channels = nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # Pixel shuffle expects the input in the format (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        if self.dim_scale == 2:
            x = self.pixel_shuffle(x)
            x = self.adjust_channels(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            share_dim=1,
            drop_path_rate=0.,
            intv=2,
    ):
        super().__init__()
        self.dim = dim
        self.intv = intv
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            self.blocks.append(
                ActiveBlock(dim,
                            mlp_ratio=mlp_ratio,
                            drop_path=dpr[i],
                            share_dim=share_dim,
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            new_offset=(i % self.intv == 0 and i != depth - 1),
                            )
            )

    def forward(self, x):
        for j, blk in enumerate(self.blocks):
            if j % self.intv == 0 and j != len(self.blocks) - 1:
                # generate new offset
                # x = self.pos_block(x)
                x, offset = blk(x)
            else:
                # forward with old offset
                x = blk(x, offset)

        return x


class Layer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            # output_dim=64,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            share_dim=1,
            upsample=None,
            drop_path_rate=0.,
            intv=2,
    ):
        super().__init__()
        self.input_dim = dim
        self.norm = nn.LayerNorm(dim)
        self.block = Block(dim=dim // 4,
                           depth=depth,
                           share_dim=share_dim,
                           mlp_ratio=mlp_ratio,
                           act_layer=act_layer,
                           norm_layer=norm_layer,
                           drop_path_rate=drop_path_rate,
                           intv=intv)
        self.proj = nn.Linear(dim, dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        if upsample is not None:
            self.upsample = PatchExpand(dim, dim_scale=2, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, H, W, C = x.shape
        assert C == self.input_dim

        x_norm = self.norm(x)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=-1)
        x_ATM1 = self.block(x1) + self.skip_scale * x1
        x_ATM2 = self.block(x2) + self.skip_scale * x2
        x_ATM3 = self.block(x3) + self.skip_scale * x3
        x_ATM4 = self.block(x4) + self.skip_scale * x4
        x_ATM = torch.cat([x_ATM1, x_ATM2, x_ATM3, x_ATM4], dim=-1)
        x_ATM = self.norm(x_ATM)
        out = self.proj(x_ATM)
        if self.upsample is not None:
            out = self.upsample(out)

        return out



class Channel_Focus_Connector_Block(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)

        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)

        return att1, att2, att3


class Spatial_Focus_Connector_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3):
        t_list = [t1, t2, t3]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2]


class MS_Att_Aggregation(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Focus_Connector_Block(c_list, split_att=split_att)
        self.satt = Spatial_Focus_Connector_Block()

    def forward(self, t1, t2, t3):
        r1, r2, r3 = t1, t2, t3

        satt1, satt2, satt3 = self.satt(t1, t2, t3)
        t1, t2, t3 = satt1 * t1, satt2 * t2, satt3 * t3

        r1_, r2_, r3_ = t1, t2, t3
        t1, t2, t3 = t1 + r1, t2 + r2, t3 + r3

        catt1, catt2, catt3 = self.catt(t1, t2, t3)
        t1, t2, t3 = catt1 * t1, catt2 * t2, catt3 * t3

        a = t1 + r1_
        b = t2 + r2_
        c = t3 + r3_
        return [a.permute(0, 2, 3, 1), b.permute(0, 2, 3, 1), c.permute(0, 2, 3, 1)]


class SubPixelConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int):
        super(SubPixelConv2d, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
        # self.upsample = PatchExpand(dim, in_channels, dim_scale=2, norm_layer=nn.LayerNorm)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(in_channels, in_channels, kernel_size=3)

    def forward(self, x, res):
        # x = self.upsample(x)
        x = x.permute(0, 3, 1, 2)
        res = res.permute(0, 3, 1, 2)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x.permute(0, 2, 3, 1)



class Decoder(nn.Module):
    def __init__(self,
                 num_classes=6,
                 depths=[2, 2, 4, 2],
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 share_dims=[1, 1, 1, 1],
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 intv=2,
                 ):
        super(Decoder, self).__init__()
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.num_layers = len(depths)
        base_dims = embed_dims[0]
        for i_layer in range(self.num_layers):
            concat_linear = WF(embed_dims[self.num_layers - 1 - i_layer]) if i_layer > 0 else nn.Identity()
            layer = Layer(dim=embed_dims[self.num_layers - 1 - i_layer],
                          depth=depths[self.num_layers - 1 - i_layer],
                          share_dim=share_dims[self.num_layers - 1 - i_layer],
                          mlp_ratio=mlp_ratios[self.num_layers - 1 - i_layer],
                          act_layer=act_layer,
                          norm_layer=norm_layer,
                          upsample=None if (i_layer == self.num_layers - 1) else PatchExpand,
                          drop_path_rate=drop_path_rate,
                          intv=intv)
            self.layers_up.append(layer)
            self.concat_back_dim.append(concat_linear)
        self.norm_up = norm_layer(embed_dims[0])

    def forward(self, x, x_downsample):
        for inx, layer in enumerate(self.layers_up):
            if inx == 0:
                x = layer(x)
            else:
                x = self.concat_back_dim[inx](x, x_downsample[3 - inx])
                x = layer(x)

        x = self.norm_up(x)  # B H W C
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CAUNet(nn.Module):
    def __init__(self,
                 drop_path_rate=0.,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 num_classes=6,
                 split_att='fc',
                 ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(num_classes=num_classes, depths=[2, 2, 4, 2], embed_dims=encoder_channels,
                               mlp_ratios=[4, 4, 4, 4], share_dims=[1, 1, 1, 1], drop_path_rate=drop_path_rate, intv=2)
        self.output = nn.Conv2d(in_channels=encoder_channels[0], out_channels=num_classes, kernel_size=1, bias=False)
        self.scab = MS_Att_Aggregation(encoder_channels, split_att)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        x = res4.permute(0, 2, 3, 1)
        x_downsample = self.scab(res1, res2, res3)
        x = self.decoder(x, x_downsample)
        x = x.permute(0, 3, 1, 2)
        x = self.output(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

