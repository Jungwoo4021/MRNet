import math
import torch
import torch.nn as nn

class MLFANet(nn.Module):
    def __init__(self, in_channels, C, embedding_size, hidden_num):
        super(MLFANet, self).__init__()
        
        self.w = nn.Parameter(torch.ones(1, 1, hidden_num, 1, 1))
        
        self.C = C
        self.norm = nn.InstanceNorm1d(in_channels * 2)
        
        # fst conv
        self.dropout = nn.Dropout()
        self.fst_conv = nn.Conv1d(in_channels * 2, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.fst_conv_bn = nn.BatchNorm1d(C)
        
        # feature processing layers
        dilations = [2, 3, 4]
        self.conv_blocks = nn.ModuleList([Bottle2neck(C, C * 3, 3, dilations[i], 8, 3) for i in range(3)])
        self.reprocessing_convs = nn.ModuleList()
        for _ in range(3):
            self.reprocessing_convs.append(
                nn.Sequential(
                    nn.Conv1d(2 * C, C, kernel_size=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(C)
                )
            )

        self.r_global_conv = nn.Sequential(
            nn.Conv1d(3 * C, C, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(C)
        )
        
        self.l_conv = nn.Sequential(
            nn.Conv1d(3 * C, C, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(C),
        )
        
        # short (s), long (l), reprocessed short (r) -term selection
        self.w_all = nn.Parameter(torch.ones(1, 7, 1))
        
        # attention
        self.att = nn.ModuleList([ASPLayer(C, C // 16) for _ in range(7)])
        
        # pooling conv
        self.bn5 = nn.BatchNorm1d(C * 7 * 2)
        self.fc6 = nn.Linear(C * 7 * 2, embedding_size)
        self.bn6 = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        # weighted sum
        x = x * self.w
        x = x.sum(2)
        x = x.permute(0, 1, 3, 2) # 
        batch, channel, hidden, time = x.size()
        x = x.reshape(batch, channel * hidden, time)
        
        # norm
        x = self.norm(x)
        
        # fst conv
        x = self.dropout(x)
        x = self.fst_conv(x)
        x = self.relu(x)
        x = self.fst_conv_bn(x)
        
        # conv blocks
        s = []
        l = []
        r = []
        for block in self.conv_blocks:
            _x = x
            x = self.dropout(x)
            x = block(x)
            x1, x2, x3 = x[:, :self.C, :], x[:, self.C:self.C * 2, :], x[:, self.C * 2:, :]
            s.append(x1)
            l.append(x2)
            r.append(x3)
            x = _x + x2
        
        # short-term feature
        
        # long-term feature
        l = torch.cat(l, dim=1)
        r_global = self.r_global_conv(l)
        l = self.l_conv(l)
        
        # reprocessed short-term feature
        for i in range(len(r)):
            r[i] = self.reprocessing_convs[i](torch.cat((r[i], r_global), dim=1))

        # pooling
        x = s + [l] + r
        x = [self.att[i](x[i]) for i in range(len(x))]
        x = torch.stack(x, dim=1)
        x = x * self.w_all
        
        x = x.view(x.size(0), -1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ASPLayer(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(ASPLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channel),
            nn.Tanh(),
            nn.Conv1d(hidden_channel, in_channel, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):
        w = self.attention(x)
        
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        
        return x

class MLFA_SEModule(nn.Module):
    def __init__(self, channels, bottleneck, num_se_module):
        super(MLFA_SEModule, self).__init__()
        self.num_se_module = num_se_module
        channels = channels // num_se_module
        bottleneck = bottleneck // num_se_module
        self.se_modules = nn.ModuleList([self.init_se_module(channels, bottleneck) for _ in range(num_se_module)])
        
    def init_se_module(self, channels, bottleneck):
        se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        return se
        
    def forward(self, input):
        buffer = []
        batch, channel, time = input.size()
        x = input.reshape(batch, self.num_se_module, channel // self.num_se_module, time)
        for i in range(self.num_se_module):
            buffer.append(self.se_modules[i](x[:, i, :, :]))
        x = torch.stack(buffer, dim=1)
        x = x.reshape(batch, channel, 1)
        
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale, num_se_module):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1

        bns = []
        convs = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = MLFA_SEModule(planes, planes // 16, num_se_module)

        self.residual_conv = None if inplanes == planes else nn.Conv1d(inplanes, planes, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x_split = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = x_split[i] if i == 0 else sp + x_split[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            x = sp if i == 0 else torch.cat((x, sp), 1)
        x = torch.cat((x, x_split[self.nums]), 1)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        
        x = self.se(x)
        
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)
            
        x += identity
        return x 