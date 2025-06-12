import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

class basicblock(nn.Module):
    def __init__(self,in_ch,out_ch,stride,attentin,frame):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch,out_ch,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(out_ch,out_ch,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm3d(out_ch)
        )
        self.relu = nn.ReLU()

        self.extra = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.extra = nn.Sequential(
                nn.Conv3d(in_ch,out_ch,kernel_size=1,stride=stride),
                nn.BatchNorm3d(out_ch)
            )

        self.attentin = attentin
        if self.attentin:
            self.attentinmap = TS_attention(out_ch, out_ch, num_heads=4,num_frames=frame)

    def forward(self,x):

        residual = x
        x = self.conv(x)
        x = self.conv1(x)
        out = self.extra(residual) + x
        if self.attentin:
            out = self.attentinmap(out) + out
        out = self.relu(out)

        return out

class blocks(nn.Module):
    def __init__(self,in_ch,out_ch,stride,num,attention,frame):

        super().__init__()

        self.block = nn.Sequential(
            *[basicblock(
                in_ch,out_ch,stride,attentin=False,frame = frame
            )] + [
                basicblock(
                    out_ch,out_ch,stride,attentin=attention,frame=frame
                ) for _ in range(num -1)
            ]
        )

    def forward(self,x):
        return self.block(x)

class Embed(nn.Module):
    def __init__(self,in_ch,embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch,embed_dim,kernel_size=1,stride=1)

    def forward(self,x):
        B,C,T,H,W = x.shape
        x = rearrange(x,'b c t h w -> (b t) c h w')
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)

        return x,T,H,W

class Time_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True,attention_type = "Spatial"):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.attention_type = attention_type
        if self.with_qkv:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_y = nn.Dropout(attn_drop)

    def forward(self, x,y = None):

        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if y is not None:
            q_y = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn_y = (q_y @ k.transpose(-2, -1)) * self.scale
            attn_y = attn_y.softmax(dim=-1)
            attn_y = self.attn_drop_y(attn_y)
            x = (attn_y @ v).transpose(1, 2).reshape(B, N, C)

        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)

        return x

class Spatial_Attention(nn.Module):
    def __init__(self, dim, heads=4, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp'):

        super(Spatial_Attention,self).__init__()

        self.reduce_size = reduce_size
        self.projection = projection
        self.scale = dim ** (-0.5)
        self.heads = heads

        self.qkv = nn.Conv2d(dim,dim*3,kernel_size=1,stride=1)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, stride=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,x,y = None):

        B, C, H, W = x.shape
        dim_head = C // self.heads

        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3,dim=1)

        if self.projection == "interp" and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))
        elif self.projection == "maxpool" and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size = self.reduce_size), (k, v))

        q = rearrange(q,'b (dim_head heads) H W -> b heads (H W) dim_head',dim_head = dim_head,heads = self.heads,H = H,W = W)
        k,v = map(lambda t: rearrange(t,'d (dim_head heads) h w -> d heads (h w) dim_head',dim_head = dim_head,heads = self.heads),(k, v))

        qk_att = torch.einsum('bhid,bhjd -> bhij',q,k)

        qk_att = qk_att * self.scale
        qk_att = F.softmax(qk_att,dim=-1)
        qk_att = self.attn_drop(qk_att)

        out = torch.einsum('bhij,bhjd -> bhid',qk_att,v)

        if y is not None:
            q_y = self.q(y)
            q_y = rearrange(q_y,'b (dim_head heads) H W -> b heads (H W) dim_head',dim_head = dim_head,heads = self.heads,H = H,W = W)
            qk_att_y = torch.einsum('bhid,bhjd -> bhij',q_y,k)
            qk_att_y = qk_att_y * self.scale
            qk_att_y = F.softmax(qk_att_y,dim=-1)
            qk_att_y = self.attn_drop(qk_att_y)
            out = torch.einsum('bhij,bhjd -> bhid',qk_att_y,v)
        out = rearrange(out,'b heads (h w) dim_head -> b (dim_head heads) h w',dim_head = dim_head,heads = self.heads,h=H,w=W)
        out = self.proj_drop(out)

        return out

class TS_attention(nn.Module):

    def __init__(self, in_ch, embed_dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_rate = 0.0, num_frames = 16, norm_layer=nn.LayerNorm):
        super().__init__()

        self.embed = Embed(in_ch,embed_dim)

        self.attn = Spatial_Attention(
           embed_dim, heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.attn_y = Spatial_Attention(
           embed_dim, heads=num_heads, attn_drop=attn_drop, proj_drop=drop)

        ## Positional Embeddings
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        ## Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(embed_dim)
        self.temporal_norm1_y = norm_layer(embed_dim)
        self.temporal_attn = Time_Attention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_attn_y = Time_Attention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(embed_dim, embed_dim)
        self.temporal_fc_y = nn.Linear(embed_dim, embed_dim)

    def forward_features(self, x):
        B = x.shape[0]
        x, T, H, W = self.embed(x)

        ## Time Embeddings
        x = x[:,:]
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)

        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)

        return x, B, T, H, W

    def forward(self, x, y = None):

        if y is None:
            x, B, T, H, W = self.forward_features(x)
            ## Temporal
            xt = x[:,:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_attn(self.temporal_norm1(xt))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,:,:] + res_temporal

            ## Spatial
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) m h w',b=B,h=H,w=W,t=T)
            res_spatial = self.attn(xs)
            res_spatial = rearrange(res_spatial, '(b t) m h w -> b (h w t) m', b=B, h=H, w=W, t=T)

            x = xt + res_spatial
            x = rearrange(x, 'b (n t) m -> b m t n', b=B, t=T)
            x = rearrange(x, 'b m t (h w) -> b m t h w', b=B, t=T,h = H,w = W)

            return x
        else:
            x, B, T, H, W = self.forward_features(x)
            y, _, _, _, _ = self.forward_features(y)
            ## Temporal
            xt = x[:,:,:]
            yt = y[:, :, :]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            yt = rearrange(yt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
            res_temporal_x = self.temporal_attn(self.temporal_norm1(xt),self.temporal_norm1_y(yt))
            res_temporal_x = rearrange(res_temporal_x, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal_x = self.temporal_fc(res_temporal_x)
            xt = x[:,:,:] + res_temporal_x
            res_temporal_y = self.temporal_attn(self.temporal_norm1_y(yt),self.temporal_norm1(xt))
            res_temporal_y = rearrange(res_temporal_y, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal_y = self.temporal_fc_y(res_temporal_y)
            yt = y[:,:,:] + res_temporal_y

            ## Spatial
            xs = xt
            ys = yt
            xs = rearrange(xs, 'b (h w t) m -> (b t) m h w',b=B,h=H,w=W,t=T)
            ys = rearrange(ys, 'b (h w t) m -> (b t) m h w', b=B, h=H, w=W, t=T)
            res_spatial_x = self.attn(xs,ys)
            res_spatial_x = rearrange(res_spatial_x, '(b t) m h w -> b (h w t) m', b=B, h=H, w=W, t=T)

            x = xt + res_spatial_x
            x = rearrange(x, 'b (n t) m -> b m t n', b=B, t=T)
            x = rearrange(x, 'b m t (h w) -> b m t h w', b=B, t=T,h = H,w = W)

            res_spatial_y = self.attn_y(ys,xs)
            res_spatial_y = rearrange(res_spatial_y, '(b t) m h w -> b (h w t) m', b=B, h=H, w=W, t=T)

            y = yt + res_spatial_y
            y = rearrange(y, 'b (n t) m -> b m t n', b=B, t=T)
            y = rearrange(y, 'b m t (h w) -> b m t h w', b=B, t=T,h = H,w = W)

            return x, y

class fusion_in_CEUS(nn.Module):
    def __init__(self,in_ch,out_ch,kernel,stride,padding):
        super().__init__()

        self.CBR_H = nn.Sequential(
            nn.Conv3d(in_ch,out_ch,kernel,stride,padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

        self.CBR_L = nn.Sequential(
            nn.Conv3d(in_ch,out_ch,kernel,stride,padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

        self.CBR_D = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride, padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

        self.CBR_FINAL = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride, padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):

        L,H = torch.chunk(x,2,dim=2)
        diff = H - L
        multi = torch.mul(self.CBR_H(H),self.CBR_D(diff))
        out =  multi + H
        out = torch.cat((self.CBR_L(L),out),dim = 2)
        out = self.CBR_FINAL(out)

        return out

class classific_model(nn.Module):
    def __init__(self,in_ch,base_ch,num_class,num_frame,block_layer,attention_apply,fusion_apply):
        super().__init__()
        self.num_blocks = len(block_layer)
        self.num_fusion = len(fusion_apply)
        self.fusion_apply = [num *2 for num in fusion_apply]

        self.first_ceus = nn.Sequential(
            nn.Conv3d(in_ch,base_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU()
        )

        self.first_bmodel = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU()
        )

        self.layers_bmodel = nn.ModuleList()
        b_ch = base_ch
        frame_bmodel = num_frame
        for i,num in enumerate(block_layer):
            self.layers_bmodel.append(
            blocks(b_ch,b_ch*2,stride=1,num = 2,attention=attention_apply[i],frame=int(frame_bmodel))
            )
            b_ch *= 2
            if i != self.num_blocks-1:
                self.layers_bmodel.append(
                    nn.MaxPool3d(kernel_size=2,stride=2)
                )
                frame_bmodel /= 2
            elif i == self.num_blocks-1:
                self.layers_bmodel.append(
                    nn.AdaptiveAvgPool3d((1,1,1))
                )
        self.layers_ceus = nn.ModuleList()
        self.layers_fusion = nn.ModuleList()
        ceus_ch = base_ch
        for i,num in enumerate(block_layer):
            self.layers_ceus.append(
            blocks(ceus_ch,ceus_ch*2,stride=1,num = 2,attention=False,frame=num_frame)
            )
            if i in fusion_apply:
                self.layers_fusion.append(
                    fusion_in_CEUS(ceus_ch*2,ceus_ch*2,kernel=3,stride=1,padding=1)
                )
            ceus_ch *= 2
            if i != self.num_blocks-1:
                self.layers_ceus.append(
                    nn.MaxPool3d(kernel_size=2,stride=2)
                )
            elif i == self.num_blocks-1:
                self.layers_ceus.append(
                    nn.AdaptiveAvgPool3d((1,1,1))
                )
        self.cross_fusion = nn.ModuleList()
        self.cross_fusion.append(
            TS_attention(ceus_ch, ceus_ch, num_heads=4, num_frames=int(frame_bmodel))
        )
        self.ave = nn.AdaptiveAvgPool3d((1,1,1))

        self.fn_bmodel = nn.Linear(b_ch, num_class)
        self.fn_ceus = nn.Linear(b_ch, num_class)
        self.fn_cross = nn.Linear(b_ch*2, num_class)
        self.all = nn.Linear(b_ch,num_class)

    def forward(self,ceus,bmodel):

        ceus = self.first_ceus(ceus)
        bmodel = self.first_bmodel(bmodel)

        bmodel_f = []
        ceus_f = []

        for i,layer in enumerate(self.layers_bmodel):
            bmodel = layer(bmodel)
            if i % 2 == 0:
                bmodel_f.append(bmodel)

        for i,layer in enumerate(self.layers_ceus):
            ceus = layer(ceus)
            if i in self.fusion_apply:
                fusions = self.layers_fusion[i//2-1]
                ceus = fusions(ceus)
            if i % 2 == 0:
                ceus_f.append(ceus)
        # fusion_feature1,fusion_feature2 = self.cross_fusion[0](ceus_f[-1],bmodel_f[-1])
        # feature = torch.cat([fusion_feature1,fusion_feature2],dim=1)
        # feature = self.ave(feature)
        # cross = self.fn_cross(feature.view(feature.size(0),-1))
        bmodel = self.fn_bmodel(bmodel.view(bmodel.size(0),-1))
        ceus = self.fn_ceus(ceus.view(ceus.size(0),-1))
        # all = cross + bmodel + ceus
        all = bmodel + ceus
        return bmodel,ceus,all

if __name__ == '__main__':
    x = torch.randn((1, 3, 32, 128, 128))
    y = torch.randn((1, 3, 32, 128, 128))
    model = classific_model(3,8,2,32,[2,2,2,2],[False,False,False,False],[])
    # model = TS_attention(3,24,12)
    # a,b,cross,all = model(x,y)
    a, b, all = model(x, y)
    print(a.shape,b.shape)
    # params_vector = torch.nn.utils.parameters_to_vector(model.parameters())
    # num_params = params_vector.numel()
    # print(num_params)