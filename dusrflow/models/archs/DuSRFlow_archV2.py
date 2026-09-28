import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ResList, PixelShufflePack, Res_Attention_Conf, ResBlock
from .matching import CenterMatching, KernelFreeMatching, PatchWarping
from .SISR import SISR_block
from .flow_2E import Flow_Net 
from .arch_util import FlowGuidedDCN, tensor_shift
from .dcn import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
from .network_utils import *

class FlowGuidedAlign(nn.Module):

    def __init__(self, nf=64, groups=8):
        super(FlowGuidedAlign, self).__init__()
        self.offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True) 
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ref_fea2X, ref_fea2X_flow, lr_shake_fea, flows):
        offset = torch.cat([ref_fea2X_flow, lr_shake_fea, flows], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))
        fea = self.lrelu(self.dcnpack(ref_fea2X, offset, flows))
        return fea
    
class MlF_extract(nn.Module):
    def __init__(self, n_feats=64):
        super(MlF_extract, self).__init__()
        
        # First convolution stage; preserves the channel dimension.
        self.stage1 = ResBlock(in_channels = n_feats, out_channels = n_feats)
        self.stage1_down = nn.Conv2d(n_feats, n_feats, 3, stride=2, padding=1)
        # Second convolution stage; increases the channel dimension.
        self.stage2 = ResBlock(n_feats, n_feats)


    def forward(self, x):
        f1 = self.stage1(x)  # Shape: B x 64 x H/2 x W/2.
        
        f2 = self.stage2(self.stage1_down(f1))  # Shape: B x 128 x H/4 x W/4.
        return f1, f2

class MlF_extract_LR(nn.Module):
    def __init__(self, n_feats=64):
        super(MlF_extract_LR, self).__init__()
        self.upsample = PixelShufflePack(n_feats, n_feats, 2, upsample_kernel=3)
        self.normsample= nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1)

    def forward(self, x):
        f1 = self.upsample(x) 
        f2 = self.normsample(x)
        return f1, f2
class DuSRFlow(nn.Module):
    def __init__(self, args):

        self.args = args
        super(DuSRFlow, self).__init__()
        self.patch_phase = getattr(args, 'patch_phase', 'even')

        self.threshold_alpha = 20
        n_feats = 64
        self.avgpool_2 = nn.AvgPool2d((2,2),(2,2))
        self.avgpool_4 = nn.AvgPool2d((4,4),(4,4))

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #SISR
        self.SISR = SISR_block()
        self.MlF_extract_LR = MlF_extract_LR(n_feats)
        self.lr_upsample = PixelShufflePack(n_feats, n_feats, 2, upsample_kernel=3)

        #lr_nearby encoder
        self.lr_nearby_head = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1))
        # self.lr_nearby_encoder = ResList(4, n_feats)
        self.upsample_lr_nearby = PixelShufflePack(n_feats, n_feats, 2, upsample_kernel=3)

        #ref encoder
        self.ref_head = nn.Sequential(nn.Conv2d(3, n_feats, 3, 1, 1))
        # self.ref_encoder = ResList(4, n_feats)
        self.MlF_extract = MlF_extract(n_feats)
        # DuFlowNet and flow-guided DCN
        self.duflownet = Flow_Net(getattr(args,'flownet_weight', False))
        
        self.duflownet.requires_grad_(True)
        self.Ref_align = FlowGuidedAlign(nf=64, groups=8)

        #kernel-free matching  and warping
        # self.center_match = CenterMatching(stride=1)
        self.kf_matching = KernelFreeMatching(stride=1)
        self.patch_warping = PatchWarping(scale=4, ksize=4)
                                                                                               
        
        #alpha Likely of limited utility
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(1)
        self.alphaConf2X = nn.Sequential(nn.Conv2d(1, n_feats//4, 7, 1, 3),
                                         self.lrelu,
                                         nn.Conv2d(n_feats//4, n_feats, 3, 1, 1),
                                         self.lrelu)

        #AdaFuison
        self.AdaFuison = Res_Attention_Conf(n_feats*2, n_feats*2, res_scale=1, SA=True, CA=True)
        self.fusion_tail = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats, 3, 1, 1),
            self.lrelu
        )
        #c2
        self.kf_dcn_warping_lv1 = KFDCNWarping(n_feats)
        self.kf_dcn_warping_lv2 = KFDCNWarping(n_feats)
        self.lv2_upsample = PixelShufflePack(n_feats, n_feats//2, 2, upsample_kernel=3)
        self.ref_fusion = nn.Sequential(
            nn.Conv2d(n_feats+n_feats//2, n_feats, 3, 1, 1),
            self.lrelu,
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            self.lrelu,
        )
        #decoder
        self.decoder = ResList(2, n_feats)
        self.decoder_tail = nn.Sequential(nn.Conv2d(n_feats, n_feats//2, 3, 1, 1),
                                         self.lrelu,
                                         nn.Conv2d(n_feats//2, 3, 3, 1, 1))

    def forward(self, lr, lr_nearby, ref, HR):  #lr(q), lr_nearby, that is lr_center(k), ref(v)
        if self.training:
            lr_fea, ref_fea_flow_align = self.flow_warping(lr, lr_nearby, ref)

            # center_confidence_map, center_index_map, center_index_map_sqr = self.center_match(lr, lr_nearby) #Kernel-Free matching

            ref_fea_flow_align_lv1,ref_fea_flow_align_lv2 = self.MlF_extract(ref_fea_flow_align)
            lr_fea_lv1, lr_fea_lv2 = self.MlF_extract_LR(lr_fea)
            ref_fea_warped_corner, corner_confidence_map, corner_index_map = self.multi_matching(
                                                                lr, lr_nearby, 
                                                                lr_fea_lv1, lr_fea_lv2,
                                                                ref_fea_flow_align_lv1, ref_fea_flow_align_lv2)
            
            ref_fea_warped_center = self.patch_warping(lr, corner_index_map, ref_fea_flow_align)
            
            out = self.fusion_decoder(corner_confidence_map, ref_fea_warped_corner, ref_fea_warped_center, self.lr_upsample(lr_fea))

            return out
        
        else:
            lr_fea, ref_fea_flow_align = self.flow_warping(lr, lr_nearby, ref)
            ref_fea_flow_align_lv1, ref_fea_flow_align_lv2 = self.MlF_extract(ref_fea_flow_align)
            lr_fea_lv1, lr_fea_lv2 = self.MlF_extract_LR(lr_fea)
            #The image has been padded 8 pixels
            H, W = lr.shape[2], lr.shape[3]

            h, w = lr_nearby.shape[2], lr_nearby.shape[3]
            ph = h //2
            pw = w //2
            delta = 8

            assert h %2 == 0 or w %2 == 0 
            

            start_hs = []
            end_hs = []
            start_ws = []
            end_ws = []        

            sr_list = []
            for i in range(3):
                for j in range(3):
                    if j != 2:
                        start_h = ph*j + delta*j
                        end_h = ph*(j+2)+ delta*j
                    else:
                        start_h = H - 2*ph
                        end_h = H                      
                    if i != 2:
                        start_w = pw*i + delta*i
                        end_w = pw*(i+2)+ delta*i
                    else:
                        start_w = W - 2*pw
                        end_w = W 

                    if self.patch_phase == 'even':
                        if start_w %2 != 0 :
                            start_w = start_w - 1
                            end_w = end_w - 1 
                        if start_h %2 != 0 :
                            start_h = start_h - 1 
                            end_h = end_h - 1

                    # assert start_w %2 == 0 and start_h %2 == 0 

                    start_hs.append(start_h*2)
                    start_ws.append(start_w*2)
                    end_hs.append(end_h*2)
                    end_ws.append(end_w*2)

                    lr_patch = lr[..., start_h:end_h, start_w:end_w]
                    lr_fea_patch = lr_fea[..., start_h:end_h, start_w:end_w]

                    lr_fea_lv2_patch = lr_fea_lv2[..., start_h : end_h, start_w:end_w]
                    lr_fea_lv1_patch = lr_fea_lv1[..., start_h*2 : end_h*2, start_w*2:end_w*2]

                    # center_confidence_map, center_index_map, center_index_map_sqr = self.center_match(lr_patch, lr_nearby, 0) #Kernel-Free matching
                    ref_fea_warped_corner, corner_confidence_map, corner_index_map = self.multi_matching(
                                                                lr_patch, lr_nearby, 
                                                                lr_fea_lv1_patch, lr_fea_lv2_patch, 
                                                                ref_fea_flow_align_lv1, ref_fea_flow_align_lv2, 
                                                                1)
 
                    ref_fea_warped_center = self.patch_warping(lr_patch, corner_index_map, ref_fea_flow_align)
                    
                    patch_sr = self.fusion_decoder(corner_confidence_map, ref_fea_warped_corner, ref_fea_warped_center, self.lr_upsample(lr_fea_patch))

                    sr_list.append(patch_sr[:,:,16:-16, 16:-16])
                    # sr_list.append(patch_sr[:,:,:, :])

            # sr_list = torch.cat(sr_list, dim=0)
            # sr_list = sr_list.view(sr_list.shape[0],-1)
            # sr_list = sr_list.permute(1,0)  #torch.Size([1, 248832, 140])
            # sr_list = torch.unsqueeze(sr_list, 0)  #torch.Size([1, 248832, 140])
            # output = F.fold(sr_list, output_size=(H*2, W*2), kernel_size=(2*ph,2*pw), padding=0, stride=(2*ph,2*pw))
            output = stitch_with_overlap_torch(sr_list, H, W, delta, start_hs, start_ws, end_hs, end_ws, HR)
            return output
        
    def multi_matching(self,lr, lr_nearby, lr_fea_lv1, lr_fea_lv2, ref_fea_flow_align_lv1, ref_fea_flow_align_lv2, row=False):
        corner_confidence_map, corner_index_map, corner_index_map_sqr = self.kf_matching(lr, lr_nearby, row) #Kernel-Free matching

        ref_fea_warped_corner_lv1 = self.kf_dcn_warping_lv1(corner_index_map_sqr, lr_fea_lv1, ref_fea_flow_align_lv1, scale=4)
        corner_confidence_map_lv1= F.interpolate(corner_confidence_map, scale_factor=4, mode='bicubic')
        ref_fea_warped_corner_lv1 = ref_fea_warped_corner_lv1 *  corner_confidence_map_lv1

        ref_fea_warped_corner_lv2 = self.kf_dcn_warping_lv2(corner_index_map_sqr, lr_fea_lv2, ref_fea_flow_align_lv2, scale=2)
        ref_fea_warped_corner_lv2 = ref_fea_warped_corner_lv2 * F.interpolate(corner_confidence_map, scale_factor=2, mode='bicubic')
        ref_fea_warped_corner_lv2 = self.lv2_upsample(ref_fea_warped_corner_lv2)



        ref_fea_warped_corner = self.ref_fusion(torch.cat([
                ref_fea_warped_corner_lv1, 
                ref_fea_warped_corner_lv2 
                # ref_fea_warped_corner_lv3
                ],dim=1))
        
        return ref_fea_warped_corner, corner_confidence_map_lv1, corner_index_map
    def flow_warping(self, lr, lr_nearby, ref):
        lr_fea = self.SISR(lr)
        lr_nearby_fea_up = self.upsample_lr_nearby(self.lr_nearby_head(lr_nearby))
        ref_fea = self.ref_head(ref)

        ref_down  = self.avgpool_2(ref)
        flows = self.duflownet(lr_nearby, ref_down)
        flows_up = F.interpolate(input=flows, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        ref_fea_flow = backward_warp(ref_fea, flows_up)#
        ref_fea_flow_align = self.Ref_align(ref_fea, ref_fea_flow, lr_nearby_fea_up, flows_up)
        return lr_fea, ref_fea_flow_align
    
    def fusion_decoder(self, corner_confidence_map, ref_fea_warped_corner, ref_fea_warped_center, lr_fea_up):
        corner_confidence_map = self.alphaConf2X(corner_confidence_map)
        
        ref_fea_warped = ref_fea_warped_corner * (1. - corner_confidence_map) + ref_fea_warped_center * corner_confidence_map

        cat_fea = torch.cat((ref_fea_warped , lr_fea_up), 1)
        fused_fea = self.fusion_tail(self.AdaFuison(cat_fea))

        out = self.decoder(fused_fea)

        out = self.decoder_tail(out + lr_fea_up)
        return out


class KFDCNWarping(nn.Module):
    def __init__(self, n_feats):
        super(KFDCNWarping, self).__init__()
        self.small_offset_conv1 = nn.Conv2d(
            n_feats*3, n_feats, 3, 1, 1, bias=True)  # concat for diff
        self.small_offset_conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)
        self.small_dyn_agg = DynAgg(
            n_feats,
            n_feats,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=8,
            extra_offset_mask=True,
            use_sim=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def index_to_flow(self, max_idx):
        device = max_idx.device
        # max_idx to flow
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device),
            torch.arange(0, w).to(device))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h),
                        dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid  # shape:(1, w, h, 2)
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

        return flow

    def flow_warp(self,
                  x,
                  flow,
                  interp_mode='bilinear',
                  padding_mode='zeros',
                  align_corners=True):
        """Warp an image or feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2), normal value.
            interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
            padding_mode (str): 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Before pytorch 1.3, the default value is
                align_corners=True. After pytorch 1.3, the default value is
                align_corners=False. Here, we use the True as default.
        Returns:
            Tensor: Warped image or feature map.
        """

        assert x.size()[-2:] == flow.size()[1:3]
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).type_as(x),
            torch.arange(0, w).type_as(x))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False

        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x,
                               vgrid_scaled,
                               mode=interp_mode,
                               padding_mode=padding_mode,
                               align_corners=align_corners)
        return output  
    
    def forward(self, corner_index_map_sqr, lr_fea, ref_fea_flow_align, scale=2):
        batch_offset = []
        batch_pre_fea= []
        for ind in range(lr_fea.size(0)):

            offset = self.index_to_flow(corner_index_map_sqr[ind])
            offset = offset[:,1:-1,1:-1,:]

            flow = torch.repeat_interleave(offset, scale, 1)
            flow = torch.repeat_interleave(flow, scale, 2)
            flow *= scale

            pre_swapped_feat = self.flow_warp(ref_fea_flow_align[ind:ind+1], flow)
            # shift offset relu3

            offset = torch.repeat_interleave(offset, scale, 1)
            # Match the offset resolution to the reference feature resolution.
            offset = torch.repeat_interleave(offset, scale, 2)
            offset *= scale
            # shift offset relu1
            shifted_offset = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset, (i * scale, j * scale))
                    shifted_offset.append(flow_shift)
                    
            shifted_offset = torch.cat(shifted_offset, dim=0)
            batch_offset.append(shifted_offset)
            batch_pre_fea.append(pre_swapped_feat)
        batch_offset_stack = torch.stack(batch_offset, dim=0)
        batch_pre_fea_stack = torch.cat(batch_pre_fea, dim=0)
        relu3_offset = torch.cat([lr_fea, batch_pre_fea_stack, ref_fea_flow_align], 1)
        relu3_offset = self.lrelu(self.small_offset_conv1(relu3_offset))
        relu3_offset = self.lrelu(self.small_offset_conv2(relu3_offset))

        ref_fea_warped_corner = self.lrelu(
            self.small_dyn_agg([ref_fea_flow_align, relu3_offset],  batch_offset_stack))
        return ref_fea_warped_corner
    
def stitch_with_overlap_torch(patches, H, W, delta, start_hs, start_ws, end_hs, end_ws, HR):
    """
    Merge nine (B, C, 2H, 2W) patches into a (B, C, 4H, 4W) feature map.
    ``patches`` follows row-major order:
             [ [0:2H,0:2W], [H:3H,0:2W], [2H:4H,0:2W],
               [0:2H,W:3W], [H:3H,W:3W], [2H:4H,W:3W],
               [0:2H,2W:4W], [H:3H,2W:4W], [2H:4H,2W:4W] ]
    H and W are half of the original input feature size.
    """
    
    device = patches[0].device
    B, C, _, _ = patches[0].shape

    out = torch.zeros((B, C, 2*(H-2*delta), 2*(W-2*delta)), device=device)
    count = torch.zeros((B, 1, 2*(H-2*delta), 2*(W-2*delta)), device=device)

    # coords = [
    #     (start_hs[0], end_hs[0], start_ws[0], end_ws[0]),
    #     (start_hs[1], end_hs[0], start_ws[0], end_ws[0]),
    #     (start_hs[2], end_hs[0], start_ws[0], end_ws[0]),

    #     (start_hs[3], , end_hs[0], start_ws[0], end_ws[0]),
    #     (start_hs[4], 3*H - delta, W + delta, 3*W - delta),
    #     (start_hs[5], 4*H, W + delta, 3*W - delta),

    #     (start_hs[6], 2*H - 2*delta, 2*W + 2*delta, 4*W),
    #     (start_hs[7], 3*H - delta, 2*W + 2*delta, 4*W),
    #     (start_hs[8], 4*H, 2*W + 2*delta, 4*W),
    # ]

    for patch, y1, y2, x1, x2 in zip(patches, start_hs, end_hs, start_ws, end_ws):
        y2 = y2 - 4*delta
        x2 = x2 - 4*delta
        out[:, :, y1:y2, x1:x2] += patch
        count[:, :, y1:y2, x1:x2] += 1
   
    out /= torch.clamp(count, min=1e-8)  # Avoid division by zero.
    return out

if __name__ == "__main__":
    pass
