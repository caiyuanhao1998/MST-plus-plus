import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# ----------------------------------------
#               Conv2d Block
# ----------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False, scale_factor=2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type,
                                  activation, norm, sn)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv2d(x)
        return x


class ResConv2dLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='lrelu',
                 norm='none', sn=False, scale_factor=2):
        super(ResConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm,
                        sn),
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation='none',
                        norm=norm, sn=sn)
        )

    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = 0.1 * out + residual
        return out


class DenseConv2dLayer_5C(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(DenseConv2dLayer_5C, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels + latent_channels, latent_channels, kernel_size, stride, padding, dilation,
                                 pad_type, activation, norm, sn)
        self.conv3 = Conv2dLayer(in_channels + latent_channels * 2, latent_channels, kernel_size, stride, padding,
                                 dilation, pad_type, activation, norm, sn)
        self.conv4 = Conv2dLayer(in_channels + latent_channels * 3, latent_channels, kernel_size, stride, padding,
                                 dilation, pad_type, activation, norm, sn)
        self.conv5 = Conv2dLayer(in_channels + latent_channels * 4, in_channels, kernel_size, stride, padding, dilation,
                                 pad_type, activation, norm, sn)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(ResidualDenseBlock_5C, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels + latent_channels, latent_channels, kernel_size, stride, padding, dilation,
                                 pad_type, activation, norm, sn)
        self.conv3 = Conv2dLayer(in_channels + latent_channels * 2, latent_channels, kernel_size, stride, padding,
                                 dilation, pad_type, activation, norm, sn)
        self.conv4 = Conv2dLayer(in_channels + latent_channels * 3, latent_channels, kernel_size, stride, padding,
                                 dilation, pad_type, activation, norm, sn)
        self.conv5 = Conv2dLayer(in_channels + latent_channels * 4, in_channels, kernel_size, stride, padding, dilation,
                                 pad_type, activation, norm, sn)

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = 0.1 * x5 + residual
        return x5


# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)  # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# ----------------------------------------
#             Non-local Block
# ----------------------------------------
class Self_Attn(nn.Module):
    """ Self attention Layer for Feature Map dimension"""

    def __init__(self, in_dim, latent_dim=8):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // latent_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // latent_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key = self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x N x N, N = H x W
        energy = torch.bmm(proj_query, proj_key)
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, height, width)

        out = self.gamma * out + x
        return out


# ----------------------------------------
#              Global Block
# ----------------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GlobalBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero', activation='lrelu',
                 norm='none', sn=False, reduction=8):
        super(GlobalBlock, self).__init__()
        self.conv1 = Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation,
                                 norm, sn)
        self.conv2 = Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation,
                                 norm, sn)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # residual
        residual = x
        # Sequeeze-and-Excitation(SE)
        b, c, _, _ = x.size()
        x = self.conv1(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = self.conv2(x)
        # addition
        out = 0.1 * y + residual
        return out
def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    downscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]
    kernel = torch.zeros(size = [downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                        device = input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride = downscale_factor, groups = c)

class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        downscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                 Generator
# ----------------------------------------
class SGN(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, start_channels=64, pad='zero', activ='lrelu', norm='none', ):
        super(SGN, self).__init__()
        # Top subnetwork, K = 3
        self.top1 = Conv2dLayer(in_channels * (4 ** 3), start_channels * (2 ** 3), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.top21 = ResidualDenseBlock_5C(start_channels * (2 ** 3), start_channels * (2 ** 2), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.top22 = GlobalBlock(start_channels * (2 ** 3), 3, 1, 1, pad_type = pad, activation = activ, norm = norm, sn = False, reduction = 4)
        self.top3 = Conv2dLayer(start_channels * (2 ** 3), start_channels * (2 ** 3), 1, 1, 0, pad_type = pad, activation = activ, norm = norm)
        # Middle subnetwork, K = 2
        self.mid1 = Conv2dLayer(in_channels * (4 ** 2), start_channels * (2 ** 2), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.mid2 = Conv2dLayer(int(start_channels * (2 ** 2 + 2 ** 3 / 4)), start_channels * (2 ** 2), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.mid31 = ResidualDenseBlock_5C(start_channels * (2 ** 2), start_channels * (2 ** 1), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.mid32 = GlobalBlock(start_channels * (2 ** 2), 3, 1, 1, pad_type = pad, activation = activ, norm = norm, sn = False, reduction = 4)
        self.mid4 = Conv2dLayer(start_channels * (2 ** 2), start_channels * (2 ** 2), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        # Bottom subnetwork, K = 1
        self.bot1 = Conv2dLayer(in_channels * (4 ** 1), start_channels * (2 ** 1), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.bot2 = Conv2dLayer(int(start_channels * (2 ** 1 + 2 ** 2 / 4)), start_channels * (2 ** 1), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.bot31 = ResidualDenseBlock_5C(start_channels * (2 ** 1), start_channels * (2 ** 0), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.bot32 = ResidualDenseBlock_5C(start_channels * (2 ** 1), start_channels * (2 ** 0), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.bot33 = GlobalBlock(start_channels * (2 ** 1), 3, 1, 1, pad_type = pad, activation = activ, norm = norm, sn = False, reduction = 4)
        self.bot4 = Conv2dLayer(start_channels * (2 ** 1), start_channels * (2 ** 1), 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        # Mainstream
        self.main1 = Conv2dLayer(in_channels, start_channels, 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.main2 = Conv2dLayer(int(start_channels * (2 ** 0 + 2 ** 1 / 4)), start_channels, 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.main31 = ResidualDenseBlock_5C(start_channels, start_channels // 2, 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.main32 = ResidualDenseBlock_5C(start_channels, start_channels // 2, 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.main33 = ResidualDenseBlock_5C(start_channels, start_channels // 2, 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.main34 = ResidualDenseBlock_5C(start_channels, start_channels // 2, 3, 1, 1, pad_type = pad, activation = activ, norm = norm)
        self.main35 = GlobalBlock(start_channels, 3, 1, 1, pad_type = pad, activation = activ, norm = norm, sn = False, reduction = 4)
        self.main4 = Conv2dLayer(start_channels, out_channels, 3, 1, 1, pad_type = pad, activation = activ, norm = norm)

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        # PixelUnShuffle                                        input: batch * 3 * 256 * 256
        x1 = pixel_unshuffle(x, 2)               # out: batch * 12 * 128 * 128
        x2 = pixel_unshuffle(x, 4)               # out: batch * 48 * 64 * 64
        x3 = pixel_unshuffle(x, 8)               # out: batch * 192 * 32 * 32
        # Top subnetwork                                        suppose the start_channels = 32
        x3 = self.top1(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top21(x3)                                     # out: batch * 256 * 32 * 32
        x3 = self.top22(x3)                                     # out: batch * 256 * 32 * 32
        x3 = self.top3(x3)                                      # out: batch * 256 * 32 * 32
        x3 = F.pixel_shuffle(x3, 2)                             # out: batch * 64 * 64 * 64, ready to be concatenated
        # Middle subnetwork
        x2 = self.mid1(x2)                                      # out: batch * 128 * 64 * 64
        x2 = torch.cat((x2, x3), 1)                             # out: batch * (128 + 64) * 64 * 64
        x2 = self.mid2(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid31(x2)                                     # out: batch * 128 * 64 * 64
        x2 = self.mid32(x2)                                     # out: batch * 128 * 64 * 64
        x2 = self.mid4(x2)                                      # out: batch * 128 * 64 * 64
        x2 = F.pixel_shuffle(x2, 2)                             # out: batch * 32 * 128 * 128, ready to be concatenated
        # Bottom subnetwork
        x1 = self.bot1(x1)                                      # out: batch * 64 * 128 * 128
        x1 = torch.cat((x1, x2), 1)                             # out: batch * (64 + 32) * 128 * 128
        x1 = self.bot2(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot31(x1)                                     # out: batch * 64 * 128 * 128
        x1 = self.bot32(x1)                                     # out: batch * 64 * 128 * 128
        x1 = self.bot33(x1)                                     # out: batch * 64 * 128 * 128
        x1 = self.bot4(x1)                                      # out: batch * 64 * 128 * 128
        x1 = F.pixel_shuffle(x1, 2)                             # out: batch * 16 * 256 * 256, ready to be concatenated
        # U-Net generator with skip connections from encoder to decoder
        x = self.main1(x)                                       # out: batch * 32 * 256 * 256
        x = torch.cat((x, x1), 1)                               # out: batch * (32 + 16) * 256 * 256
        x = self.main2(x)                                       # out: batch * 32 * 256 * 256
        x = self.main31(x)                                      # out: batch * 32 * 256 * 256
        x = self.main32(x)                                      # out: batch * 32 * 256 * 256
        x = self.main33(x)                                      # out: batch * 32 * 256 * 256
        x = self.main34(x)                                      # out: batch * 32 * 256 * 256
        x = self.main35(x)                                      # out: batch * 32 * 256 * 256
        x = self.main4(x)                                       # out: batch * 3 * 256 * 256

        return x[:, :, :h_inp, :w_inp]

