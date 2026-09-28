import torch
import torch.nn as nn
from .common import MeanShift
from torchvision import models
import torch.nn.functional as F


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks
     
class CenterMatching(nn.Module):
    def __init__(self, stride=1):
        super(CenterMatching, self).__init__()

        self.stride = stride  

        self.feature_extract = nn.AvgPool2d(2, stride=2)
    

    def forward(self, query, key, byRow=False):

        # query = self.sub_mean(query)
        # key = self.sub_mean(key)

        query_fea = self.feature_extract(query)
        shape_query_fea = query_fea.shape
        query_fea_unfold = extract_image_patches(query_fea, ksizes=[3, 3], strides=[self.stride,self.stride], rates=[1, 1], padding='same')
        key_fea = self.feature_extract(key)
        key_fea_unflod = extract_image_patches(key_fea, ksizes=[3, 3], strides=[self.stride, self.stride], rates=[1, 1], padding='same')

        key_fea_unflod = key_fea_unflod.permute(0, 2, 1)    
        key_fea_unflod = F.normalize(key_fea_unflod, dim=2) #torch.Size([4, 4096, 144])
        query_fea_unfold  = F.normalize(query_fea_unfold, dim=1) #torch.Size([4, 144, 4096])

        if byRow:
            relavance_maps = torch.zeros((query_fea_unfold.shape[0], query_fea_unfold.shape[2]), device=query_fea_unfold.device)
            hard_indices = torch.zeros((query_fea_unfold.shape[0], query_fea_unfold.shape[2]), device=query_fea_unfold.device).to(torch.int64)

            w = shape_query_fea[3]
            for h in range(shape_query_fea[2]):
                q_line = query_fea_unfold[:,:, h*w:(h+1)*w]
                sim = torch.bmm(key_fea_unflod, q_line)
                relavance_lines, hard_indices_lines = torch.max(sim, dim=1) 
                relavance_maps[:,   h*w:(h+1)*w] = relavance_lines
                hard_indices[:,   h*w:(h+1)*w] = hard_indices_lines 
        else:
            y = torch.bmm(key_fea_unflod, query_fea_unfold) 
            relavance_maps, hard_indices = torch.max(y, dim=1) 

        relavance_maps = relavance_maps.view(shape_query_fea[0], 1, shape_query_fea[2], shape_query_fea[3])

        hard_indices_sqr =  hard_indices.view(shape_query_fea[0], shape_query_fea[2], shape_query_fea[3])

        return relavance_maps,  hard_indices,  hard_indices_sqr

class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTE, self).__init__()
        
        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(7, 12):
            # self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            # for param in self.slice3.parameters():
                # param.requires_grad = requires_grad
        
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        
        self.conv = nn.Conv2d(128, 4, 3, 1, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        
        x_lv2 = self.conv(x_lv2)
        # x = self.slice3(x)
        # x_lv3 = x
        return x_lv1, x_lv2 #, x_lv3
  
class KernelFreeMatching(nn.Module):
    def __init__(self, stride=1):
        super(KernelFreeMatching, self).__init__()

        self.stride = stride  
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.feature_extract = LTE()
    
    def matching(self, query_fea, key_fea, byRow=False):
        shape_query_fea = query_fea.shape
        query_fea_unfold = extract_image_patches(query_fea, ksizes=[3, 3], strides=[self.stride,self.stride], rates=[1, 1], padding='same')

        key_fea_unflod = extract_image_patches(key_fea, ksizes=[3, 3], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        key_fea_unflod = key_fea_unflod.permute(0, 2, 1)    

        key_fea_unflod = F.normalize(key_fea_unflod, dim=2) #torch.Size([4, 4096, 144])
        query_fea_unfold  = F.normalize(query_fea_unfold, dim=1) #torch.Size([4, 144, 4096])

        if byRow:
            relavance_maps = torch.zeros((query_fea_unfold.shape[0], query_fea_unfold.shape[2]), device=query_fea_unfold.device)
            hard_indices = torch.zeros((query_fea_unfold.shape[0], query_fea_unfold.shape[2]), device=query_fea_unfold.device).to(torch.int64)

            w = shape_query_fea[3]
            for h in range(shape_query_fea[2]):
                q_line = query_fea_unfold[:,:, h*w:(h+1)*w]
                sim = torch.bmm(key_fea_unflod, q_line)
                relavance_lines, hard_indices_lines = torch.max(sim, dim=1) 
                relavance_maps[:,   h*w:(h+1)*w] = relavance_lines
                hard_indices[:,   h*w:(h+1)*w] = hard_indices_lines 
        else:
            y = torch.bmm(key_fea_unflod, query_fea_unfold) 
            relavance_maps, hard_indices = torch.max(y, dim=1) 
        relavance_maps = relavance_maps.view(shape_query_fea[0], 1, shape_query_fea[2], shape_query_fea[3])
        hard_indices_sqr =  hard_indices.view(shape_query_fea[0], shape_query_fea[2], shape_query_fea[3])
        
        return relavance_maps,  hard_indices, hard_indices_sqr
    def forward(self, query, key, byRow=False):

        query_fea_lv1, query_fea_lv2 = self.feature_extract(query)
        key_fea_lv1, key_fea_lv2 = self.feature_extract(key)

        relavance_maps_lv2,  hard_indices_lv2, hard_indices_sqr_lv2 = self.matching(query_fea_lv2, key_fea_lv2, byRow)

        return relavance_maps_lv2, hard_indices_lv2,  hard_indices_sqr_lv2


class PatchWarping(nn.Module):
    def __init__(self,  scale, ksize):
        super(PatchWarping, self).__init__()
        self.ksize = ksize     #3
        self.scale = scale

    def warp(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lr, index_map, value):

        shape_out = list(lr.size()) 

        unfolded_value = extract_image_patches(value, ksizes=[self.ksize , self.ksize ],  strides=[self.scale,self.scale], rates=[1, 1], padding='same')
        warpped_value = self.warp(unfolded_value, 2, index_map) 
        warpped_features = self.fold_with_normalization(warpped_value, output_size=(shape_out[2]*2, shape_out[3]*2), kernel_size=self.ksize, padding=0, stride=self.scale) 
     
        return warpped_features     
    def fold_with_normalization(self, patches, output_size, kernel_size, padding, stride):
        ones = torch.ones_like(patches)
        weight = F.fold(ones, output_size, kernel_size=kernel_size, padding=padding, stride=stride)
        image = F.fold(patches, output_size, kernel_size=kernel_size, padding=padding, stride=stride)
        return image / (weight + 1e-8)
    
