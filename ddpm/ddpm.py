import torch
import torch.nn as nn
from einops import rearrange

class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.sample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        x = self.sample(x)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels):
        super(Downsample, self).__init__()
        self.sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        x = self.sample(x)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super(ResidualBlock, self).__init__()
        if in_channels != out_channels:
            self.convx = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.convx = nn.Identity()
        self.convt = nn.Conv2d(time_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x, t):
        xi = self.convx(x)
        x = self.group_norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        t = self.convt(t)
        x = x + t
        x = self.group_norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = x + xi
        return x


class Diffusion(nn.Module):
    def __init__(self, T, img_ch, beta_range = [0.0001, 0.02],  embedding_size = 32, channels = [1, 2], blocks = 2):
        super(Diffusion, self).__init__()
        self.block = blocks
        self.img_ch = img_ch
        self.embedding = nn.Embedding(T, embedding_size)
        self.convx = nn.Conv2d(self.img_ch, embedding_size, kernel_size=3, padding=1, bias=False)
        time_channels = embedding_size * 4
        self.convt = nn.Conv2d(embedding_size, time_channels, kernel_size=3, padding=1, bias=False)
        
        self.down_blocks = nn.ModuleList()
        in_ch = embedding_size
        skip_chans = []
        for i, ch in enumerate(channels):
            for _ in range(blocks):
                self.down_blocks.append(ResidualBlock(in_ch, ch * embedding_size, time_channels))
                in_ch = ch * embedding_size
            skip_chans.append(in_ch)
            self.down_blocks.append(Downsample(in_ch))   
        
        self.up_blocks = nn.ModuleList()
        for i, ch in enumerate(channels[::-1]):
            for _ in range(blocks):
                self.up_blocks.append(ResidualBlock(in_ch, ch * embedding_size, time_channels))
                in_ch = ch * embedding_size
            self.up_blocks.append(Upsample(in_ch))
            in_ch += skip_chans.pop()

        self.conv2 = nn.Conv2d(in_ch, self.img_ch, kernel_size=3, padding=1, bias=False)
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.activation = nn.SiLU()


        beta = torch.linspace(*beta_range, T, dtype=torch.float32)
        alpha = 1.0 - beta
        bar_alpha = torch.cumprod(alpha, 0)
        sqrt_bar_alpha = torch.sqrt(bar_alpha)
        sqrt_bar_beta = torch.sqrt(1 - bar_alpha)
        self.register_buffer('sqrt_bar_alpha', sqrt_bar_alpha)
        self.register_buffer('sqrt_bar_beta', sqrt_bar_beta)


    def sample(self, x_0, t):
        noise = torch.randn_like(x_0).to(x_0.device)
        shape = (x_0.shape[0], 1, 1, 1)
        return x_0 * self.sqrt_bar_alpha[t].view(shape) + noise * self.sqrt_bar_beta[t].view(shape), noise

    def forward(self, x, t):
        t = self.embedding(t)
        t = rearrange(t, 'b c -> b c () ()')
        x = self.convx(x) # [B, C, H, W]
        t = self.activation(self.convt(t)) # [B, 4*C, H, W]
        
        inputs = []
        for block in self.down_blocks: 
            if isinstance(block, Downsample):
                inputs.append(x) 
            x = block(x, t)
        
        for block in self.up_blocks:
            x = block(x, t)
            if isinstance(block, Upsample):
                x = torch.cat([x, inputs.pop()], dim=1)

        x = self.group_norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x
    

if __name__ == "__main__":
    T = 1000
    model = Diffusion(T)
    x = torch.randn(2, 3, 128, 128)
    t = torch.randint(0, T, (2,))
    y = model(x, t)
    print(y.shape)