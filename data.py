import glob
import random
import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        transforms_ = [ transforms.Resize(int(143), Image.BICUBIC), 
                transforms.RandomCrop(128), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
              ]
        #content source
        self.transform = transforms.Compose(transforms_)
        self.X = sorted(glob.glob(os.path.join(root, f'{mode}Content', '*')))
        
        #style image source(s)
        self.Y = []
        style_sources = sorted(glob.glob(os.path.join(root, f'{mode}Styles', '*')))
        for label,style in enumerate(style_sources):
            temp = [(label,x) for x in sorted(glob.glob(style_sources[label]+"/*"))]
            self.Y.extend(temp)
        
        
    def __getitem__(self,index):
                                                 
        output = {}
        output['content'] = self.transform(Image.open(self.X[index % len(self.X)]))
        
        #select style
        selection = self.Y[random.randint(0, len(self.Y) - 1)]
        
        try:                                         
            output['style'] = self.transform(Image.open(selection[1]))
        except:
            print('thisuns grey')
            print(selection)
                        
        output['style_label'] = selection[0]
    
        return output
    
    def __len__(self):
        return max(len(self.X), len(self.Y))

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))