import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ResList, PixelShufflePack, Res_Attention_Conf, ResBlock
from .matching import KernelFreeMatching, PatchWarping
from .SISR import SISR_block
from .flow_2E import Flow_Net
from .arch_util import FlowGuidedDCN, tensor_shift
from .dcn import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
from .network_utils import backward_warp


class FlowGuidedAlignment(nn.Module):

    def __init__(self, nf=64, groups=8):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dcnpack = FlowGuidedDCN(
            nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ref_fea, warped_ref_fea, lrc_fea_up, flows):
        offset = torch.cat([warped_ref_fea, lrc_fea_up, flows], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))
        return self.lrelu(self.dcnpack(ref_fea, offset, flows))


class MultiLevelRefFeatureExtractor(nn.Module):
    def __init__(self, n_feats=64):
        super().__init__()

        self.stage1 = ResBlock(in_channels=n_feats, out_channels=n_feats)
        self.stage1_down = nn.Conv2d(n_feats, n_feats, 3, stride=2, padding=1)
        self.stage2 = ResBlock(n_feats, n_feats)

    def forward(self, x):
        f1 = self.stage1(x)

        f2 = self.stage2(self.stage1_down(f1))
        return f1, f2


class MultiLevelLRFeatureExtractor(nn.Module):
    def __init__(self, n_feats=64):
        super().__init__()
        self.upsample = PixelShufflePack(n_feats, n_feats, 2, upsample_kernel=3)
        self.normsample = nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1)

    def forward(self, x):
        f1 = self.upsample(x)
        f2 = self.normsample(x)
        return f1, f2


class DuSRFlow(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.inference_matching_row = getattr(args, "inference_matching_row", 1)
        n_feats = 64
        self.avgpool_2 = nn.AvgPool2d((2, 2), (2, 2))

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # SISR encoder
        self.SISR = SISR_block()
        self.MlF_extract_LR = MultiLevelLRFeatureExtractor(n_feats)
        self.lr_upsample = PixelShufflePack(n_feats, n_feats, 2, upsample_kernel=3)

        # LRC encoder
        self.lr_nearby_head = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1))
        self.upsample_lr_nearby = PixelShufflePack(
            n_feats, n_feats, 2, upsample_kernel=3
        )

        # Ref encoder
        self.ref_head = nn.Sequential(nn.Conv2d(3, n_feats, 3, 1, 1))
        self.MlF_extract = MultiLevelRefFeatureExtractor(n_feats)

        # Dual-lens flow-based alignment
        self.duflownet = Flow_Net(getattr(args, "flownet_weight", False))

        self.duflownet.requires_grad_(True)
        self.Ref_align = FlowGuidedAlignment(nf=64, groups=8)

        # KF matching and patch warping
        self.kf_matching = KernelFreeMatching(stride=1)
        self.patch_warping = PatchWarping(scale=4, ksize=4)

        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.alphaConf2X = nn.Sequential(
            nn.Conv2d(1, n_feats // 4, 7, 1, 3),
            self.lrelu,
            nn.Conv2d(n_feats // 4, n_feats, 3, 1, 1),
            self.lrelu,
        )

        # LR-Ref fusion
        self.AdaFuison = Res_Attention_Conf(
            n_feats * 2, n_feats * 2, res_scale=1, SA=True, CA=True
        )
        self.fusion_tail = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1), self.lrelu
        )
        # DCN warping
        self.kf_dcn_warping_lv1 = KFDCNWarping(n_feats)
        self.kf_dcn_warping_lv2 = KFDCNWarping(n_feats)
        self.lv2_upsample = PixelShufflePack(
            n_feats, n_feats // 2, 2, upsample_kernel=3
        )
        self.ref_fusion = nn.Sequential(
            nn.Conv2d(n_feats + n_feats // 2, n_feats, 3, 1, 1),
            self.lrelu,
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            self.lrelu,
        )
        # Reconstruction decoder
        self.decoder = ResList(2, n_feats)
        self.decoder_tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, 3, 1, 1),
            self.lrelu,
            nn.Conv2d(n_feats // 2, 3, 3, 1, 1),
        )

    def forward(self, lr, lr_nearby, ref):
        """Run the DuSRFlow forward pass.

        Args:
            lr: Wide-angle low-resolution input.
            lr_nearby: Wide-angle center crop (LRC) corresponding to Ref.
            ref: Telephoto reference image.
        """
        # SISR Encoder
        lr_fea = self.sisr_encoder(lr)

        # Dual-Lens Flow-Based Alignment
        ref_fea_aligned = self.dual_lens_flow_based_alignment(lr_nearby, ref)

        if self.training:
            # KF Matching
            matching_results = self.kernel_free_matching(lr, lr_nearby)

            # Complementary Reference Warping
            ref_fea_warped = self.complementary_reference_warping(
                lr,
                lr_fea,
                ref_fea_aligned,
                matching_results,
            )

            # LR-Ref Fusion
            return self.lr_ref_fusion(ref_fea_warped, self.lr_upsample(lr_fea))

        else:
            # Cache full-image features once for tiled inference.
            ref_warping_features = self.MlF_extract(ref_fea_aligned)
            lr_warping_features = self.MlF_extract_LR(lr_fea)
            ref_fea_aligned_lv1, ref_fea_aligned_lv2 = ref_warping_features
            lr_fea_lv1, lr_fea_lv2 = lr_warping_features
            # Remove the eight-pixel context padding on each side after tiling.
            H, W = lr.shape[2] - 16, lr.shape[3] - 16

            h, w = lr_nearby.shape[2], lr_nearby.shape[3]

            ph = h - 16
            pw = w - 16

            num_x = W // pw
            num_y = H // ph
            sr_list = []
            for j in range(num_y):
                for i in range(num_x):
                    lr_patch = lr[
                        :, :, j * (ph) : j * (ph) + ph + 16, i * pw : i * pw + pw + 16
                    ]
                    lr_fea_patch = lr_fea[
                        :, :, j * (ph) : j * (ph) + ph + 16, i * pw : i * pw + pw + 16
                    ]
                    lr_fea_lv1_patch = lr_fea_lv1[
                        :,
                        :,
                        j * ph * 2 : j * ph * 2 + ph * 2 + 32,
                        i * pw * 2 : i * pw * 2 + pw * 2 + 32,
                    ]
                    lr_fea_lv2_patch = lr_fea_lv2[
                        :, :, j * ph : j * ph + ph + 16, i * pw : i * pw + pw + 16
                    ]

                    # KF Matching
                    matching_results = self.kernel_free_matching(
                        lr_patch, lr_nearby, self.inference_matching_row
                    )

                    # Complementary Reference Warping
                    ref_fea_warped = self.complementary_reference_warping(
                        lr_patch,
                        lr_fea_patch,
                        ref_fea_aligned,
                        matching_results,
                        lr_warping_features=(lr_fea_lv1_patch, lr_fea_lv2_patch),
                        ref_warping_features=(ref_fea_aligned_lv1, ref_fea_aligned_lv2),
                    )

                    # LR-Ref Fusion
                    patch_sr = self.lr_ref_fusion(
                        ref_fea_warped, self.lr_upsample(lr_fea_patch)
                    )

                    sr_list.append(patch_sr[:, :, 16:-16, 16:-16])

            sr_list = torch.cat(sr_list, dim=0)
            sr_list = sr_list.view(sr_list.shape[0], -1)
            sr_list = sr_list.permute(1, 0)
            sr_list = torch.unsqueeze(sr_list, 0)
            output = F.fold(
                sr_list,
                output_size=(H * 2, W * 2),
                kernel_size=(2 * ph, 2 * pw),
                padding=0,
                stride=(2 * ph, 2 * pw),
            )

            return output

    def sisr_encoder(self, lr):
        """Encode the LR input with the SISR encoder."""
        return self.SISR(lr)

    def kernel_free_matching(self, lr, lr_nearby, row=False):
        """Return KF matching confidence and correspondence maps."""
        return self.kf_matching(lr, lr_nearby, row)

    def dual_lens_flow_based_alignment(self, lr_nearby, ref):
        """Align Ref to LRC with dual-lens flow-based alignment."""
        lr_nearby_fea_up = self.upsample_lr_nearby(self.lr_nearby_head(lr_nearby))
        ref_fea = self.ref_head(ref)

        ref_down = self.avgpool_2(ref)
        flows = self.duflownet(lr_nearby, ref_down)
        flows_up = (
            F.interpolate(
                input=flows, scale_factor=2, mode="bilinear", align_corners=True
            )
            * 2.0
        )
        ref_fea_flow = backward_warp(ref_fea, flows_up)
        ref_fea_flow_align = self.Ref_align(
            ref_fea, ref_fea_flow, lr_nearby_fea_up, flows_up
        )
        return ref_fea_flow_align

    def dcn_warping_branch(
        self,
        confidence_map,
        index_map_sqr,
        lr_warping_features,
        ref_warping_features,
    ):
        """DCN-warping branch of Complementary Reference Warping."""
        lr_fea_lv1, lr_fea_lv2 = lr_warping_features
        ref_fea_aligned_lv1, ref_fea_aligned_lv2 = ref_warping_features

        ref_fea_dcn_warped_lv1 = self.kf_dcn_warping_lv1(
            index_map_sqr, lr_fea_lv1, ref_fea_aligned_lv1, scale=4
        )
        confidence_map_lv1 = F.interpolate(
            confidence_map, scale_factor=4, mode="bicubic"
        )
        ref_fea_dcn_warped_lv1 = ref_fea_dcn_warped_lv1 * confidence_map_lv1

        ref_fea_dcn_warped_lv2 = self.kf_dcn_warping_lv2(
            index_map_sqr, lr_fea_lv2, ref_fea_aligned_lv2, scale=2
        )
        ref_fea_dcn_warped_lv2 = ref_fea_dcn_warped_lv2 * F.interpolate(
            confidence_map, scale_factor=2, mode="bicubic"
        )
        ref_fea_dcn_warped_lv2 = self.lv2_upsample(ref_fea_dcn_warped_lv2)

        return (
            self.ref_fusion(
                torch.cat([ref_fea_dcn_warped_lv1, ref_fea_dcn_warped_lv2], dim=1)
            ),
            confidence_map_lv1,
        )

    def patch_warping_branch(self, lr, index_map, ref_fea_aligned):
        """Patch-warping branch of Complementary Reference Warping."""
        return self.patch_warping(lr, index_map, ref_fea_aligned)

    def reference_fusion(
        self, confidence_map, ref_fea_dcn_warped, ref_fea_patch_warped
    ):
        """Confidence-aware Ref-fusion in Complementary Reference Warping."""
        confidence_map = self.alphaConf2X(confidence_map)
        return (
            ref_fea_dcn_warped * (1.0 - confidence_map)
            + ref_fea_patch_warped * confidence_map
        )

    def complementary_reference_warping(
        self,
        lr,
        lr_fea,
        ref_fea_aligned,
        matching_results,
        lr_warping_features=None,
        ref_warping_features=None,
    ):
        """Complementary Reference Warping.

        It consists of Patch-warping, DCN-warping, and confidence-aware
        Ref-fusion.
        """
        confidence_map, index_map, index_map_sqr = matching_results

        if lr_warping_features is None:
            lr_warping_features = self.MlF_extract_LR(lr_fea)
        if ref_warping_features is None:
            ref_warping_features = self.MlF_extract(ref_fea_aligned)

        ref_fea_dcn_warped, confidence_map = self.dcn_warping_branch(
            confidence_map,
            index_map_sqr,
            lr_warping_features,
            ref_warping_features,
        )
        ref_fea_patch_warped = self.patch_warping_branch(lr, index_map, ref_fea_aligned)
        return self.reference_fusion(
            confidence_map, ref_fea_dcn_warped, ref_fea_patch_warped
        )

    def lr_ref_fusion(self, ref_fea_warped, lr_fea_up):
        """Fuse LR and Ref features and reconstruct the SR output."""

        cat_fea = torch.cat((ref_fea_warped, lr_fea_up), 1)
        fused_fea = self.fusion_tail(self.AdaFuison(cat_fea))

        out = self.decoder(fused_fea)

        out = self.decoder_tail(out + lr_fea_up)
        return out


class KFDCNWarping(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.small_offset_conv1 = nn.Conv2d(n_feats * 3, n_feats, 3, 1, 1, bias=True)
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
            use_sim=False,
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def index_to_flow(self, max_idx):
        device = max_idx.device
        # Convert flattened correspondence indices into two-dimensional flow.
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device), torch.arange(0, w).to(device), indexing="ij"
        )
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h), dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

        return flow

    def flow_warp(
        self, x, flow, interp_mode="bilinear", padding_mode="zeros", align_corners=True
    ):
        """Warp an image or feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2), normal value.
            interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
            padding_mode (str): 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Grid-sampling corner alignment mode.
        Returns:
            Tensor: Warped image or feature map.
        """

        assert x.size()[-2:] == flow.size()[1:3]
        _, _, h, w = x.size()
        # Create the base sampling grid.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x), indexing="ij"
        )
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False

        vgrid = grid + flow
        # Normalize the sampling grid to [-1, 1].
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(
            x,
            vgrid_scaled,
            mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return output

    def forward(self, index_map_sqr, lr_fea, ref_fea_aligned, scale=2):
        batch_offset = []
        batch_prewarped_ref_fea = []
        for ind in range(lr_fea.size(0)):

            offset = self.index_to_flow(index_map_sqr[ind])
            offset = offset[:, 1:-1, 1:-1, :]

            flow = torch.repeat_interleave(offset, scale, 1)
            flow = torch.repeat_interleave(flow, scale, 2)
            flow *= scale

            prewarped_ref_fea = self.flow_warp(ref_fea_aligned[ind : ind + 1], flow)

            offset = torch.repeat_interleave(offset, scale, 1)
            # Match the offset resolution to the reference feature resolution.
            offset = torch.repeat_interleave(offset, scale, 2)
            offset *= scale
            shifted_offset = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset, (i * scale, j * scale))
                    shifted_offset.append(flow_shift)

            shifted_offset = torch.cat(shifted_offset, dim=0)
            batch_offset.append(shifted_offset)
            batch_prewarped_ref_fea.append(prewarped_ref_fea)
        batch_offset_stack = torch.stack(batch_offset, dim=0)
        prewarped_ref_fea = torch.cat(batch_prewarped_ref_fea, dim=0)
        learned_offset = torch.cat([lr_fea, prewarped_ref_fea, ref_fea_aligned], 1)
        learned_offset = self.lrelu(self.small_offset_conv1(learned_offset))
        learned_offset = self.lrelu(self.small_offset_conv2(learned_offset))

        ref_fea_dcn_warped = self.lrelu(
            self.small_dyn_agg([ref_fea_aligned, learned_offset], batch_offset_stack)
        )
        return ref_fea_dcn_warped
