import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################

class DownSample(nn.Module):
    def __init__(self, inChannels:int):
        super().__init__()
        self.downsample = nn.Conv1d(inChannels, inChannels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, inChannels:int, scaleFactor:int=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scaleFactor, mode="nearest"), # dim * scale_factor
            nn.Conv1d(inChannels, inChannels//2 , 3, padding=1),
        )
    
    def forward(self, x):
        return self.upsample(x)

class ResidualBlock(nn.Module):
    def __init__(self, inChannels:int, outChannels:int):
        super().__init__()

        if inChannels != outChannels:
            self.residual_connections = nn.Conv1d(inChannels, outChannels, 1)
        else:
            self.residual_connections = nn.Identity()

        self.conv1 = nn.Conv1d(inChannels, outChannels, 3, padding=1)
        self.conv2 = nn.Conv1d(outChannels, outChannels, 3, padding=1)
    
    def forward(self, x):
        residual = self.residual_connections(x)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = F.relu(out + residual)
        return out

################################################################################

class Unet(nn.Module):
    def __init__(self, inChannels:int, baseChannels:int, channelMult:tuple[int]=(1,2,4,8,16), numResBlock:int=2):
        super(Unet, self).__init__()
        self.numResBlock = numResBlock
        self.init_conv = nn.Conv1d(inChannels, baseChannels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        nowChannels = baseChannels

        # Downsampling -> Channel increase, dim decrease
        for mult in channelMult:
            outChannels = baseChannels * mult
            for _ in range(numResBlock):
                self.downs.append(ResidualBlock(nowChannels, outChannels))
                nowChannels = outChannels
            self.downs.append(DownSample(nowChannels))
        
        # Middle layer -> bridge between downsampling and upsampling
        self.mid = ResidualBlock(nowChannels, nowChannels)

        # Upsampling -> Channel decrease, dim increase
        for mult in reversed(channelMult):
            outChannels = baseChannels * mult
            self.ups.append(UpSample(nowChannels))
            nowChannels = nowChannels // 2
            for i in range(numResBlock):
                if i == 0:
                    self.ups.append(ResidualBlock(nowChannels + outChannels, outChannels))
                else:
                    self.ups.append(ResidualBlock(outChannels, outChannels))
                nowChannels = outChannels

        # Output layer
        self.out_conv = nn.Conv1d(nowChannels, inChannels, 3, padding=1)

    def forward(self, x):
        skips = []
        
        x = self.init_conv(x)
        skips.append(x)

        for i, layer in enumerate(self.downs):
            if isinstance(layer, DownSample):
                x = layer(x)
            else:
                x = layer(x)
                if i + 1 < len(self.downs) and isinstance(self.downs[i+1], DownSample):
                    skips.append(x)

        x = self.mid(x)

        for i, layer in enumerate(self.ups):
            if isinstance(layer, UpSample):
                x = layer(x)
                skip = skips.pop()
            else:
                if i % (self.numResBlock+1) == 1:
                    x = layer(torch.cat([x, skip], dim=1))
                    print("Skip")
                else:
                    x = layer(x)
        
        out = self.out_conv(x)
    
        return out

################################################################################

if __name__ == "__main__":
    model = Unet(1, 4)

    x = torch.randn(1, 1, 48000)
    x = model(x)
    print(x.shape)