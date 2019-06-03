import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight= 1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):    
        w_variance = torch.sum(torch.pow(x[:,:,:,:-1] - x[:,:,:,1:], 2))
        h_variance = torch.sum(torch.pow(x[:,:,:-1,:] - x[:,:,1:,:], 2))
        loss = self.TVLoss_weight * (h_variance + w_variance)
        return loss

#https://github.com/pytorch/pytorch/issues/9160#issuecomment-483048684
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    
#https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/models.py
class ResidualBlock(nn.Module):
    def __init__(self,in_features):
        super(ResidualBlock,self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x) #skip connection

class Encoder(nn.Module):
    def __init__(self, in_nc, ngf=64):
        super(Encoder, self).__init__()

        #Inital Conv Block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(in_nc, ngf, 7),
                    nn.InstanceNorm2d(ngf),
                    nn.ReLU(inplace=True) ]

        in_features = ngf
        out_features = in_features *2

        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]

            in_features = out_features
            out_features = in_features * 2

        self.model = nn.Sequential(*model)

    def forward(self,x):
        #Return batch w/ encoded content picture
        return [self.model(x['content']), x['style_label']]

class Transformer(nn.Module):
    def __init__(self,n_styles, ngf, auto_id=True):
        super(Transformer, self).__init__()
        self.t = nn.ModuleList([ResidualBlock(ngf*4) for i in range(n_styles)])
        if auto_id:
            self.t.append(Identity())

    def forward(self,x):
        #x0 is content, x[1][0] is label
        label = x[1][0]
        mix = np.sum([self.t[i](x[0])*v for (i,v) in enumerate(label) if v])
        #return content transformed by style specific residual block
        return mix

class Decoder(nn.Module):
    def __init__(self, out_nc, ngf, n_residual_blocks=5):
        super(Decoder, self).__init__()

        in_features = ngf * 4
        out_features = in_features//2

        model = []
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self,in_nc,out_nc,n_styles,ngf):
        super(Generator, self).__init__()

        self.encoder = Encoder(in_nc,ngf)
        self.transformer = Transformer(n_styles,ngf)
        self.decoder = Decoder(out_nc,ngf)

    def forward(self,x):
        e = self.encoder(x)
        t = self.transformer(e)
        d = self.decoder(t)
        return d

class Discriminator(nn.Module):
    """
    Patch-Gan discriminator 
    """
    def __init__(self, in_nc, n_styles, ndf=64):
        super(Discriminator, self).__init__()

        # A bunch of convolutions 
        model = [   nn.Conv2d(in_nc, 64, 4, stride=2, padding=2),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=2),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=2),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4,stride=1, padding=2),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        self.model = nn.Sequential(*model)

        # GAN (real/notreal) Output-
        self.fldiscriminator = nn.Conv2d(512, 1, 4, padding = 2)

        # Classification Output
        self.aux_clf = nn.Conv2d(512, n_styles, 4, padding = 2)

    def forward(self, x):
        base =  self.model(x)
        discrim = self.fldiscriminator(base)
        clf = self.aux_clf(base).transpose(1,3)

        return [discrim,clf]
