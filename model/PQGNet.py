import torch
from torch import nn
import torch.nn.functional as F
import pywt
from torch.autograd import Function
from einops import rearrange, repeat
import numbers
import math


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias





def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNormC(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNormC, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)


    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)




class ResidualLeakBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

        self.body = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,  kernel_size=5, stride=stride, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.proj(x)
        residual = x
        x = self.body(x)
        out = self.relu(x+residual)
        return out







# 定义一个DWT功能类，继承自Function
class DWT_Function(Function):
    # 定义前向传播静态方法
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 保证输入张量x在内存中是连续存储的
        x = x.contiguous()
        # 保存后向传播需要的参数
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        # 保存输入张量x的形状
        ctx.shape = x.shape

        # 获取输入张量x的通道数
        dim = x.shape[1]
        # 对x进行二维卷积操作，得到低频和高频分量
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 将四个分量按通道维度拼接起来
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回拼接后的结果
        return x

    # 定义反向传播静态方法
    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算x的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的权重
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            # 根据保存的形状信息重塑dx
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            # 调整dx的维度顺序并重塑
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            # 将四个小波滤波器沿零维度拼接，并重复C次以匹配输入通道数
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            # 使用转置卷积进行上采样
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        # 返回dx以及其余不需要梯度的参数
        return dx, None, None, None, None
    


# 定义一个IDWT功能类，继承自Function
class IDWT_Function(Function):
    # 定义前向传播静态方法
    @staticmethod
    def forward(ctx, x, filters):
        # 保存后向传播需要的参数
        ctx.save_for_backward(filters)
        # 保存输入张量x的形状
        ctx.shape = x.shape

        # 根据保存的形状信息调整x的形状
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        # 计算通道数
        C = x.shape[1]
        # 重塑x
        x = x.reshape(B, -1, H, W)
        # 重复滤波器C次以匹配输入通道数
        filters = filters.repeat(C, 1, 1, 1)
        # 使用转置卷积进行上采样
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        # 返回上采样的结果
        return x

    # 定义反向传播静态方法
    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算x的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的滤波器
            filters = ctx.saved_tensors
            filters = filters[0]
            # 根据保存的形状信息重塑dx
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            # 分解滤波器
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            # 对dx进行二维卷积操作，得到低频和高频分量
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            # 将四个分量按通道维度拼接起来
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回dx以及其余不需要梯度的参数
        return dx, None






# 定义一个二维逆离散小波变换模块，继承自nn.Module
class IDWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建重构低通和高通滤波器的Tensor
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        # 计算二维重构滤波器
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        # 为滤波器添加额外的维度
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        # 将四个小波滤波器沿零维度拼接
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        # 注册缓冲区变量来存储滤波器
        self.register_buffer('filters', filters)
        # 确保滤波器的数据类型为float32
        self.filters = self.filters.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用IDWT_Function的forward方法
        return IDWT_Function.apply(x, self.filters)



# 定义一个二维离散小波变换模块，继承自nn.Module
class DWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建分解低通和高通滤波器的Tensor
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        # 计算二维分解滤波器
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        # 注册缓冲区变量来存储滤波器
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # 确保滤波器的数据类型为float32
        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用DWT_Function的forward方法
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)















class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, PatchSize):
        super().__init__()
        # self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.PatchSize = PatchSize
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, out_dim , kernel_size=PatchSize, stride=PatchSize, bias=True),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )


    def forward(self, x):
        out = self.body(x)
        return out
    









class SelfAttention(nn.Module):
    def __init__(self, channel_num):
        super(SelfAttention, self).__init__()
        self.KV_size = channel_num
        self.channel_num = channel_num
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = nn.Softmax(dim=3)

        self.mheadq = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)

        self.project_out = nn.Conv2d(self.channel_num, self.channel_num, kernel_size=1, stride=1)
        self.attn_norm = LayerNormC(channel_num, LayerNorm_type='WithBias')





    def forward(self, emb_all):
        org = emb_all
        emb_all = self.attn_norm(emb_all)
        b , c, h, w = emb_all.shape
        q = self.q(self.mheadq(emb_all))
        k = self.k(self.mheadk(emb_all))
        v = self.v(self.mheadv(emb_all))


        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c ', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c ', head=self.num_attention_heads)


        # print(q.shape, k.shape, v.shape)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, num, _ = k.shape

        atten = (q @ k.transpose(-2, -1)) / math.sqrt(num)
        atten_probs = self.softmax(self.psi(atten))

        out = (atten_probs @ v)
        out = out.mean(dim=1)
        out = rearrange(out, 'b  (h w) c -> b c h w', h=h, w=w)
        out = self.project_out(out)

        cx = org + out
        org = cx

        return org


















class TransfomerDecoder(nn.Module):
    def __init__(self, qkv_dims=992, num_layers=4):
        super().__init__()
        self.qkv_dims = qkv_dims
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            layer = SelfAttention(qkv_dims)
            self.layers.append(layer)


    def forward(self, x):
        for layer_block in self.layers:
            x = layer_block(x)
        return x




class MWHL(nn.Module):
    def __init__(self, channel=32, wave = 'haar'):
        super().__init__()
        self.dwt = DWT_2D(wave)
        self.channel = channel
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv_D = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        self.ResBlock = ResidualLeakBlock(channel, channel)


    def forward(self, x):
        e_dwt = self.dwt(x)
        e_ll, e_lh, e_hl, e_hh = e_dwt.split(self.channel, 1)
        e_down = self.maxpool(x)
        e2 = self.conv_D(torch.cat([e_ll, e_down], dim=1))
        out = self.ResBlock(e2)
        return out





class PFCM(nn.Module):
    def __init__(self, dim=32, out_ch=1,feature_size=[256, 128, 64, 32, 16], num_layers=4):
        super().__init__()
        self.in_dim = dim * len(feature_size)
        self.dim = dim
        self.feature_size = feature_size
        PatchSize = feature_size[-1]
        self.PatchSize = feature_size[-1]
        self.PatchSizeLayers = []
        self.PatchEmbeddings = nn.ModuleList()
        self.qkv_dim = 0
        self.num_layers = num_layers

        for i,size in enumerate(feature_size):
            PS = size // PatchSize
            self.PatchSizeLayers.append(PS)
            self.PatchEmbeddings.append(PatchEmbedding(dim, dim, PS))
            self.qkv_dim += dim


        


        self.TransDecoder = TransfomerDecoder(qkv_dims=self.qkv_dim, num_layers=self.num_layers)

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.qkv_dim, out_channels=dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU()
        )

        self.Conv2 = nn.Conv2d(self.dim, out_ch, kernel_size=1, stride=1, padding=0)

    

    def forward(self, x):
        # 1.将多尺度的特征进行编码到Patch块的格式
        features = []
        for i, en in enumerate(x):
            f = self.PatchEmbeddings[i](en)
            features.append(f)
        features = torch.cat(features, dim=1)


        # 2.TransformerEncoder编码器
        features = self.TransDecoder(features)
        features = self.Conv1(features)
        out = self.Conv2(features)
        return out, features







class PQGM(nn.Module):
    def __init__(self, dim, feature_size, Patchnum):
        super().__init__()
        # assert feature_size%Patchnum == 0
        self.feature_size = feature_size
        self.patch_num = Patchnum
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d((Patchnum, Patchnum))

        self.q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.KV_size = dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, e, PAM_out):
        identity = e
        b , c, h, w = e.shape
        max_e = self.avg_pool(e)
        q = self.q(max_e)
        k = self.k(PAM_out)
        v = self.v(e)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attention_probs = self.softmax(attn)
        out = (attention_probs @ v)
        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)
        out = self.project_out(out) + e
        return out







class AlternateCat(nn.Module):
    def __init__(self, dim=1, num=3):
        """
        沿指定维度交替拼接两个张量。

        Args:
            dim (int, optional): 指定的维度，默认为 1（通道维度）。
        """
        super().__init__()
        self.dim = dim
        self.num = num

    def forward(self, x_list):
        """
        沿指定维度交替拼接两个张量。

        Args:
            x (torch.Tensor): 第一个输入张量。
            y (torch.Tensor): 第二个输入张量。

        Returns:
            torch.Tensor: 沿指定维度交替拼接后的张量。

        Raises:
            AssertionError: 如果输入张量在指定维度上的大小不一致。
        """
        # 确保两个张量在指定维度上的大小一致
        # assert x.shape == y.shape, f'x.shape:{x.shape} != y.shape:{y.shape}'
        assert len(x_list) == self.num, f'input num error!'
        for i in range(self.num):
            assert x_list[0].shape == x_list[i].shape, f'input index{i} shape error!'



        # 获取指定维度的大小
        size = x_list[0].size(self.dim)
        # print(size)

        x_list_slices = []
        # 将 x 和 y 沿着指定维度拆分为单个元素的切片
        for i in range(self.num):
            x_list_slices.append(torch.split(x_list[i], 1, dim=self.dim))

        # 交替拼接 x 和 y 的切片
        interleaved_slices = []
        for i in range(size):
            for j in range(self.num):
                interleaved_slices.append(x_list_slices[j][i])
            # interleaved_slices.extend([x_slices[i], y_slices[i]])

        # 沿着指定维度堆叠交替后的切片
        concatenated = torch.cat(interleaved_slices, dim=self.dim)

        return concatenated












class AlCattention(nn.Module):
    def __init__(self, dim):
        super(AlCattention, self).__init__()
        self.dim = dim  # 保存通道维度参数
        # 自适应平均池化：将空间维度压缩为1x1（保留通道维度信息）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应最大池化：同样压缩空间维度，捕捉通道维度的最大值信息
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # MLP多层感知机：用于将统计特征映射为通道权重
        self.alcat = AlternateCat(dim=1, num=3)
        self.share_mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1 , stride=1, padding=0, bias=True, groups=self.dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1 , stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.spconv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3 , stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, highf):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        std = torch.std(x, dim=(2, 3), keepdim=True)  # 空间维度标准差：(B, 2C)
        x_ams = self.alcat([x_avg, x_max, std])
        channel_weights = self.share_mlp(x_ams)
        x_1 = x * channel_weights + x



        avg_out = torch.mean(x_1, dim=1, keepdim=True)
        max_out, _ = torch.max(x_1, dim=1, keepdim=True)

        # 空间特征拼接
        spatial_features = torch.cat([avg_out,  max_out], dim=1)
        spatial_weights = self.spconv(spatial_features)
        dwt = []
        dwt.append(x_1)
        for i in range(len(highf)):
            temp = highf[i] * spatial_weights.expand_as(highf[i]) + highf[i]
            dwt.append(temp)
        dwt = torch.cat(dwt, dim=1)
        return dwt













class HEWL(nn.Module):
    def __init__(self, dim=32, out_ch=1, scale_factor=8,wave='haar'):
        super().__init__()
        self.dim = dim
        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)
        self.alca = AlCattention(dim=dim)

        self.Conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True)
        )


        self.PredOut = nn.Sequential(
            nn.Conv2d(dim, out_ch, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=scale_factor)
        )

        self.Out = ResidualLeakBlock(dim * 2, dim)



        





    
    def forward(self, x, e):
        e_dwt = self.dwt(e)
        e_ll, e_lh, e_hl, e_hh = e_dwt.split(self.dim, 1)
        e_high = [e_lh, e_hl, e_hh]
        x_idwt = self.alca(x, e_high)
        x = self.idwt(x_idwt)
        d = self.Conv(x)

        pred = self.PredOut(d)
        d = torch.cat((e, d), dim=1)
        out = self.Out(d)
        return out, pred














class PQGNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dim=32, feature_size=[256, 128, 64, 32, 16]):
        super().__init__()
        """
        args:
            in_ch: 输入通道数
            out_ch: 输出通道数
            dim: 基础通道数
            feature_size: 每个阶段的特征图大小
        """
        self.title = 'PQGNet'
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim = dim
        self.feature_size = feature_size




        self.enc1 = ResidualLeakBlock(in_ch, dim, stride=1)
        self.enc2 = MWHL(channel=dim, wave='haar')
        self.enc3 = MWHL(channel=dim, wave='haar')
        self.enc4 = MWHL(channel=dim, wave='haar')
        self.enc5 = MWHL(channel=dim, wave='haar')


        self.PFCM = PFCM(self.dim, out_ch,feature_size=[256, 128, 64, 32, 16], num_layers=2)

        self.PQGMS = nn.ModuleList()

        for i in range(len(feature_size)):
            self.PQGMS.append(
                PQGM(dim=dim, feature_size=feature_size[i], Patchnum=16)
            )


        self.dec4 = HEWL(dim=dim, out_ch=out_ch, scale_factor=8, wave='haar')
        self.dec3 = HEWL(dim=dim, out_ch=out_ch, scale_factor=4, wave='haar')
        self.dec2 = HEWL(dim=dim, out_ch=out_ch, scale_factor=2, wave='haar')
        self.dec1 = HEWL(dim=dim, out_ch=out_ch, scale_factor=1, wave='haar')


        self.Conv = nn.Conv2d(dim, out_ch, kernel_size=1, stride=1, padding=0)


    


    def forward(self, x):
        e1 = self.enc1(x)  # [B, dim, H, W]
        e2 = self.enc2(e1)  # [B, dim, H/2, W/2]
        e3 = self.enc3(e2)  # [B, dim, H/4, W/4]
        e4 = self.enc4(e3)  # [B, dim, H/8, W/8]
        e5 = self.enc5(e4)  # [B, dim, H/16, W/16]



        features = [e1, e2, e3, e4, e5]


        PQSM_out, PQSM_feature = self.PFCM(features)


        for i in range(len(self.PQGMS)):
            var_name = f'e{i + 1}'
            if var_name in locals():
                tempout = features[i]
                tempout = self.PQGMS[i](tempout, PQSM_feature)
                features[i] = tempout
        

        e1, e2, e3, e4, e5 = features


        d4, pred4 = self.dec4(e5, e4)
        d3, pred3 = self.dec3(d4, e3)
        d2, pred2 = self.dec2(d3, e2)
        d1, pred1 = self.dec1(d2, e1)

        pred0 = self.Conv(d1)


        if self.training:
            pred = []
            pred.append(pred4)
            pred.append(pred3)
            pred.append(pred2)
            pred.append(pred1)
            pred.append(pred0)
            return pred, PQSM_out
        else:
            return pred0, PQSM_out
        
from thop import profile
import time
if __name__ == "__main__":
    model = PQGNet().cuda()
    inputs = torch.rand(1, 1, 256, 256).cuda()
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')

    # 计算推理时间
    start_time = time.time()  # 记录开始时间
    with torch.no_grad():  # 不计算梯度
        for _ in range(100):  # 进行100次推理以获得更稳定的平均时间
            model(inputs)
    end_time = time.time()  # 记录结束时间
    inference_time = (end_time - start_time) / 100  # 计算平均推理时间

    print("Average Inference Time = {:.5f} seconds".format(inference_time))
