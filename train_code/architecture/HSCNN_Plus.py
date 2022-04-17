import torch.nn as nn
import torch
class dfus_block(nn.Module):
    def __init__(self, dim):
        super(dfus_block, self).__init__()
        self.conv1 = nn.Conv2d(dim, 128, 1, 1, 0, bias=False)

        self.conv_up1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_up2 = nn.Conv2d(32, 16, 1, 1, 0, bias=False)

        self.conv_down1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_down2 = nn.Conv2d(32, 16, 1, 1, 0, bias=False)

        self.conv_fution = nn.Conv2d(96, 32, 1, 1, 0, bias=False)

        #### activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        feat = self.relu(self.conv1(x))
        feat_up1 = self.relu(self.conv_up1(feat))
        feat_up2 = self.relu(self.conv_up2(feat_up1))
        feat_down1 = self.relu(self.conv_down1(feat))
        feat_down2 = self.relu(self.conv_down2(feat_down1))
        feat_fution = torch.cat([feat_up1,feat_up2,feat_down1,feat_down2],dim=1)
        feat_fution = self.relu(self.conv_fution(feat_fution))
        out = torch.cat([x, feat_fution], dim=1)
        return out

class ddfn(nn.Module):
    def __init__(self, dim, num_blocks=78):
        super(ddfn, self).__init__()

        self.conv_up1 = nn.Conv2d(dim, 32, 3, 1, 1, bias=False)
        self.conv_up2 = nn.Conv2d(32, 32, 1, 1, 0, bias=False)

        self.conv_down1 = nn.Conv2d(dim, 32, 3, 1, 1, bias=False)
        self.conv_down2 = nn.Conv2d(32, 32, 1, 1, 0, bias=False)

        dfus_blocks = [dfus_block(dim=128+32*i) for i in range(num_blocks)]
        self.dfus_blocks = nn.Sequential(*dfus_blocks)

        #### activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        feat_up1 = self.relu(self.conv_up1(x))
        feat_up2 = self.relu(self.conv_up2(feat_up1))
        feat_down1 = self.relu(self.conv_down1(x))
        feat_down2 = self.relu(self.conv_down2(feat_down1))
        feat_fution = torch.cat([feat_up1,feat_up2,feat_down1,feat_down2],dim=1)
        out = self.dfus_blocks(feat_fution)
        return out

class HSCNN_Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, num_blocks=30):
        super(HSCNN_Plus, self).__init__()

        self.ddfn = ddfn(dim=in_channels, num_blocks=num_blocks)
        self.conv_out = nn.Conv2d(128+32*num_blocks, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        fea = self.ddfn(x)
        out =  self.conv_out(fea)
        return out