import torch
import torch.nn as nn

class basicblock(nn.Module):
    def __init__(self,in_ch,out_ch,stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch,out_ch,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )
        self.conv1 = nn.Conv3d(out_ch,out_ch,kernel_size=3,stride=stride,padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()

        self.extra = nn.Sequential()

        if stride != 1 or in_ch != out_ch:
            self.extra = nn.Sequential(
                nn.Conv3d(in_ch,out_ch,kernel_size=1,stride=1),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self,x):

        residual = x
        out = self.conv(x)
        out = self.conv1(out)
        out = self.bn(out)
        out = self.extra(residual) + out
        out = self.relu(out)

        return out

class blocks(nn.Module):

    def __init__(self,in_ch,out_ch,num,stride=1):
        super().__init__()

        self.block = nn.Sequential(
            *[basicblock(
                in_ch,out_ch,stride=stride
            )] + [
                basicblock(
                    out_ch,out_ch,stride=stride
                )
            for _ in range(num - 1)
            ]
        )

    def forward(self,x):
        return self.block(x)


class fusionmodule(nn.Module):
    def __init__(self,out_ch):
        super().__init__()

        # self.conv_s = nn.Conv3d(in_ch,out_ch,kernel_size=3,stride=1,padding=1)
        # self.conv_m = nn.Conv3d(in_ch,out_ch,kernel_size=3,stride=1,padding=1)
        self.fus = nn.Sequential(
            nn.Conv3d(out_ch,out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,y,fusion = None):

        out = y - x
        if fusion is not None:
            out = torch.cat((fusion,out),dim=1)
        out = self.fus(out)

        return out

class fusionnet(nn.Module):
    def __init__(self,in_ch,base_ch,num_class,block_layer):
        super().__init__()

        self.num_blocks = len(block_layer)

        self.first_x = nn.Sequential(
            nn.Conv3d(in_ch,base_ch,kernel_size=1,stride=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU()
        )

        self.first_y = nn.Sequential(
            nn.Conv3d(in_ch,base_ch,kernel_size=1,stride=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU()
        )

        self.layers_x = nn.ModuleList()
        self.layers_y = nn.ModuleList()
        ch = []

        in_ch = base_ch
        for i,num in enumerate(block_layer):

            self.layers_x.append(
                blocks(in_ch,in_ch*2,stride=1,num=num)
            )
            self.layers_y.append(
                blocks(in_ch, in_ch * 2, stride=1, num=num)
            )

            in_ch *= 2
            ch.append(in_ch)

            if i != self.num_blocks:

                self.layers_x.append(
                    nn.MaxPool3d(kernel_size=2,stride=2)
                )
                self.layers_y.append(
                    nn.MaxPool3d(kernel_size=2, stride=2)
                )

        self.fusion = nn.ModuleList()

        fusion_ch = 0
        for i in range(self.num_blocks):
            fusion_ch += ch[i]
            self.fusion.append(
                fusionmodule(fusion_ch)
            )

            if i != self.num_blocks-1:
                self.fusion.append(
                    nn.MaxPool3d(kernel_size=2,stride=2)
                )
            else:
                self.fusion.append(
                    nn.AdaptiveAvgPool3d((1,1,1))
                )
        self.fn = nn.Linear(240,num_class)

    def forward(self,x,y):

        x = self.first_x(x)
        y = self.first_y(y)

        x_f = []
        y_f = []

        for i,layer in enumerate(self.layers_x):
            x = layer(x)
            x_f.append(x)
            #
            # if i % 2 == 0:
            #     x_f.append(x)

        for j,layer in enumerate(self.layers_y):
            y = layer(y)
            y_f.append(y)
            #
            # if j % 2 == 0:
            #     y_f.append(y)

        fusion_feature = []
        for i in range(0,len(self.fusion),2):
            xi,yi = x_f[i],y_f[i]
            layer_fus = self.fusion[i]
            if i != 0:
                out = layer_fus(xi,yi,fusion_feature.pop(0))
            else:
                out = layer_fus(xi, yi)
            layer_pool = self.fusion[i+1]
            fusion_feature.append(layer_pool(out))

        out = fusion_feature[-1].view(fusion_feature[-1].size(0),-1)
        out = self.fn(out)

        return out


if __name__ == '__main__':
    x = torch.randn((1,3,32,256,256))
    y = torch.randn((1,3,32,256,256))
    model = fusionnet(in_ch=3,base_ch=8,num_class=2,block_layer=[2,2,2,2])
    out = model(x,y)
    print(out.shape)