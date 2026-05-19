######################################## CVPR2025 BHViT start ########################################

# Shift_channel_mix 模块：
# 本研究提出了一种轻量级特征混合模块 Shift_channel_mix，旨在通过通道分割与空间偏移操作增强特征表达能力。
# 具体而言，该模块首先沿着通道维度（dim=1）对输入特征图进行四等分分块（即 x_1, x_2, x_3, x_4），随后分别在水平方向（宽度维度）和垂直方向（高度维度）上施加正负方向的循环移位（circular shift）。
# 其中，x_1 和 x_2 分别在高度方向进行正向和负向偏移，而 x_3 和 x_4 则在宽度方向进行正向和负向偏移。
# 最终，偏移后的特征块通过通道拼接（channel concatenation）重新组合，以实现跨通道的信息交互与局部特征增强。

# 该设计的核心思想是利用通道内信息重分布的方式，引导不同通道特征感受不同的空间位置信息，从而提升网络的特征表达能力。
# 此外，由于该操作仅涉及基本的通道切分与循环移位，计算复杂度极低，不引入额外的参数或显著的计算开销。
# 因此，Shift_channel_mix 适用于对计算资源受限的任务，如嵌入式视觉系统或实时目标检测等场景。
class Shift_channel_mix(nn.Module):
    def __init__(self,shift_size):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):

        x1, x2, x3, x4 = x.chunk(4, dim = 1)

        x1 = torch.roll(x1, self.shift_size, dims=2)#[:,:,1:,:]

        x2 = torch.roll(x2, -self.shift_size, dims=2)#[:,:,:-1,:]

        x3 = torch.roll(x3, self.shift_size, dims=3)#[:,:,:,1:]

        x4 = torch.roll(x4, -self.shift_size, dims=3)#[:,:,:,:-1]
         
        x = torch.cat([x1, x2, x3, x4], 1)

        return x

class EUCB_SC(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super(EUCB_SC,self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.shift_channel_mix = Shift_channel_mix(1)

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        x = self.shift_channel_mix(x)
        return x

class MSCB_SC(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[1,3,5], stride=1, expansion_factor=2, dw_parallel=True, add=True):
        super(MSCB_SC, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            Conv(self.in_channels, self.ex_channels, 1)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            Conv(self.combined_channels, self.out_channels, 1, act=False)
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        
        self.shift_channel_mix = Shift_channel_mix(1)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        x = self.shift_channel_mix(x)
        return x

class CSP_MSCB_SC(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1,3,5], shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        
        self.m = nn.ModuleList(MSCB_SC(self.c, self.c, kernel_sizes=kernel_sizes) for _ in range(n))

######################################## CVPR2025 BHViT end ########################################