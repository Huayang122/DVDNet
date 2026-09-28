# DVDNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from functools import partial

tensor_rotate = partial(rotate, interpolation=InterpolationMode.BILINEAR)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = shortcut

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        return F.relu(out)


class MultiScaleFusionGate(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=0.5):
        super(MultiScaleFusionGate, self).__init__()
        self.downsample = partial(F.interpolate, scale_factor=scale_factor, 
                                 mode='area', recompute_scale_factor=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
            
        self.fusion_core = ResidualBlock(in_channels, out_channels, shortcut=self.shortcut)

    def forward(self, x):
        x = self.downsample(x)
        return self.fusion_core(x)


class MFGM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 dilation=1, bias=False, strip_size=7):
        super(MFGM, self).__init__()
        
        self.multiscale_branch = nn.ModuleList([
            self._build_conv_block(in_channels, 24, 3, stride, 1),
            self._build_conv_block(in_channels, 24, 5, stride, 2), 
            self._build_conv_block(in_channels, 24, 7, stride, 3)
        ])
        
        self.orientation_branch = nn.ModuleList([
            nn.Conv2d(in_channels, 6, (1, strip_size), stride, padding=(0, strip_size//2)),
            nn.Conv2d(in_channels, 6, (1, strip_size//2), stride, padding=(0, strip_size//4)),
            nn.Conv2d(in_channels, 6, (strip_size, 1), stride, padding=(strip_size//2, 0)),
            nn.Conv2d(in_channels, 6, (strip_size//2, 1), stride, padding=(strip_size//4, 0))
        ])
        
        self.context_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True)
        )
        
        total_fusion_channels = 24 * 3 + 6 * 8 + 16
        
        self.feature_fusion = self._build_fusion_module(total_fusion_channels)
        
        self.refinement_net = nn.Sequential(
            nn.Conv2d(total_fusion_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def _build_conv_block(self, in_ch, out_ch, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def _build_fusion_module(self, total_channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, total_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pyramid_features = [conv(x) for conv in self.multiscale_branch]
        pyramid_out = torch.cat(pyramid_features, dim=1)
        
        directional_features = []
        
        h1 = self.orientation_branch[0](x)
        h2 = self.orientation_branch[1](x)
        directional_features.extend([h1, h2])
        
        v1 = self.orientation_branch[2](x)
        v2 = self.orientation_branch[3](x)
        directional_features.extend([v1, v2])
        
        rotated_45 = tensor_rotate(x, 45)
        d1_45 = self.orientation_branch[2](rotated_45)
        d2_45 = self.orientation_branch[3](rotated_45)
        d2_45 = tensor_rotate(d2_45, -45)
        directional_features.extend([d1_45, d2_45])
        
        rotated_135 = tensor_rotate(x, 135)
        d1_135 = self.orientation_branch[2](rotated_135)
        d2_135 = self.orientation_branch[3](rotated_135)
        d2_135 = tensor_rotate(d2_135, -135)
        directional_features.extend([d1_135, d2_135])
        
        directional_out = torch.cat(directional_features, dim=1)
        
        context_out = self.context_branch(x)
        
        all_features = torch.cat([pyramid_out, directional_out, context_out], dim=1)
        fusion_weights = self.feature_fusion(all_features)
        fused_features = all_features * fusion_weights
        
        return self.refinement_net(fused_features)


class HierarchicalEncoderStage(nn.Module):
    def __init__(self, scale_factor, in_channels=64, res_layers=64, num_blocks=3, stride=1):
        super(HierarchicalEncoderStage, self).__init__()
        self.residual_path = self._build_residual_path(in_channels, res_layers, num_blocks, stride)
        self.skip_connection = MultiScaleFusionGate(3, res_layers, scale_factor=scale_factor)

    def forward(self, input_features, residual_input):
        residual_out = self.residual_path(residual_input)
        skip_out = self.skip_connection(input_features)
        return torch.cat((residual_out, skip_out), 1)

    def _build_residual_path(self, in_channels, out_channels, num_blocks, stride):
        shortcut = None
        if in_channels != out_channels or stride != 1:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)


class LocalOptimizationDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalOptimizationDecoder, self).__init__()
        
        self.upsampling_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.upsampling_module(x)
        
        identity_upsampled = F.interpolate(
            identity, 
            size=out.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        residual = self.residual_conv(identity_upsampled)
        
        out = out + residual
        out = self.final_activation(out)
        
        return out


class GlobalOptimizationDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4, 8]):
        super(GlobalOptimizationDecoder, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.dilated_pyramid = nn.ModuleList([
            self._build_dilated_block(in_channels, in_channels // 4, dilation)
            for dilation in dilation_rates
        ])
        
        self.feature_integration = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 4 * len(dilation_rates), in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.upsampling_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _build_dilated_block(self, in_ch, out_ch, dilation):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        original_features = x
        
        pyramid_features = [dilated_conv(x) for dilated_conv in self.dilated_pyramid]
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        
        integrated_features = torch.cat([original_features, pyramid_concat], dim=1)
        fused_features = self.feature_integration(integrated_features)
        
        return self.upsampling_module(fused_features)


class PISM(nn.Module):
    def __init__(self, feature_channels=64):
        super(PISM, self).__init__()
        self.feature_channels = feature_channels
        
        self.weight_generation = nn.Sequential(
            nn.Linear(feature_channels * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        last_linear = self.weight_generation[-2]
        
        nn.init.normal_(last_linear.weight, mean=0.0, std=0.001)
        nn.init.constant_(last_linear.bias, 0.0)
        
    def forward(self, main_features, aux_features):
        main_global = F.adaptive_avg_pool2d(main_features, 1)
        aux_global = F.adaptive_avg_pool2d(aux_features, 1)
        
        global_features = torch.cat([main_global.flatten(1), aux_global.flatten(1)], dim=1)
        path_weights = self.weight_generation(global_features)
        
        return path_weights


class DVDNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, decode_mode='both'):
        super().__init__()
        
        valid_modes = ['both', 'main_only', 'aux_only']
        if decode_mode not in valid_modes:
            raise ValueError(f"decode_mode must be one of {valid_modes}, got '{decode_mode}'")
        
        self.decode_mode = decode_mode

        self.feature_extractor = MFGM(in_channels, 64, stride=2) 
        
        self.encoder_stage1 = HierarchicalEncoderStage(0.5, 64, 64, 3, 1)
        self.encoder_stage2 = HierarchicalEncoderStage(0.25, 128, 128, 4, 2)
        self.encoder_stage3 = HierarchicalEncoderStage(0.125, 256, 256, 6, 2)
        self.encoder_stage4 = HierarchicalEncoderStage(0.0625, 512, 512, 3, 2)
        
        self.fusion_gates = nn.ModuleDict({
            'gate4': MultiScaleFusionGate(1024, 512, 1),
            'gate3': MultiScaleFusionGate(512, 256, 1),
            'gate2': MultiScaleFusionGate(256, 128, 1),
            'gate1': MultiScaleFusionGate(128, 64, 1)
        })
        
        if self.decode_mode in ['both', 'main_only']:
            self.main_decoders = nn.ModuleDict({
                'decoder4': LocalOptimizationDecoder(512, 256),
                'decoder3': LocalOptimizationDecoder(256, 128),
                'decoder2': LocalOptimizationDecoder(128, 64),
                'decoder1': LocalOptimizationDecoder(64, 64)
            })
        
        if self.decode_mode in ['both', 'aux_only']:
            self.aux_decoders = nn.ModuleDict({
                'decoder4': GlobalOptimizationDecoder(512, 256),
                'decoder3': GlobalOptimizationDecoder(256, 128),
                'decoder2': GlobalOptimizationDecoder(128, 64),
                'decoder1': GlobalOptimizationDecoder(64, 64)
            })
        
        if self.decode_mode == 'both':
            self.pism = PISM(feature_channels=64)
        
        self.fusion_output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, 3, padding=1)
        )

    def _main_path_decoding(self, enc1, enc2, enc3, enc4):
        integrated4 = self.fusion_gates['gate4'](enc4)
        decoded4 = self.main_decoders['decoder4'](integrated4)
        
        enc3_aligned = self.fusion_gates['gate3'](enc3)
        decoded4 = decoded4 + enc3_aligned
        
        decoded3 = self.main_decoders['decoder3'](decoded4)
        enc2_aligned = self.fusion_gates['gate2'](enc2)
        decoded3 = decoded3 + enc2_aligned
        
        decoded2 = self.main_decoders['decoder2'](decoded3)
        enc1_aligned = self.fusion_gates['gate1'](enc1)
        decoded2 = decoded2 + enc1_aligned
        
        return self.main_decoders['decoder1'](decoded2)

    def _auxiliary_path_decoding(self, enc1, enc2, enc3, enc4):
        enc4_adjusted = self.fusion_gates['gate4'](enc4)
        auxiliary_decoded4 = self.aux_decoders['decoder4'](enc4_adjusted)
        
        enc3_adjusted = self.fusion_gates['gate3'](enc3)
        auxiliary_input3 = auxiliary_decoded4 + enc3_adjusted
        
        auxiliary_decoded3 = self.aux_decoders['decoder3'](auxiliary_input3)
        enc2_adjusted = self.fusion_gates['gate2'](enc2)     
        auxiliary_input2 = auxiliary_decoded3 + enc2_adjusted
        
        auxiliary_decoded2 = self.aux_decoders['decoder2'](auxiliary_input2)
        enc1_adjusted = self.fusion_gates['gate1'](enc1)
        auxiliary_input1 = auxiliary_decoded2 + enc1_adjusted
        
        return self.aux_decoders['decoder1'](auxiliary_input1)

    def _dual_path_fusion(self, enc1, enc2, enc3, enc4):
        main_output = self._main_path_decoding(enc1, enc2, enc3, enc4)
        auxiliary_output = self._auxiliary_path_decoding(enc1, enc2, enc3, enc4)
        
        path_weights = self.pism(main_output, auxiliary_output)
        
        w_main = path_weights[:, 0].view(-1, 1, 1, 1)
        w_auxiliary = path_weights[:, 1].view(-1, 1, 1, 1)
        
        return main_output * w_main + auxiliary_output * w_auxiliary

    def _select_decoding_path(self, enc1, enc2, enc3, enc4):
        if self.decode_mode == 'main_only':
            return self._main_path_decoding(enc1, enc2, enc3, enc4)
        elif self.decode_mode == 'aux_only':
            return self._auxiliary_path_decoding(enc1, enc2, enc3, enc4)
        else:
            return self._dual_path_fusion(enc1, enc2, enc3, enc4)

    def forward(self, x):
        initial_features = self.feature_extractor(x)
        
        encoded1 = self.encoder_stage1(x, initial_features)
        encoded2 = self.encoder_stage2(x, encoded1)
        encoded3 = self.encoder_stage3(x, encoded2)
        encoded4 = self.encoder_stage4(x, encoded3)
        
        decoded_output = self._select_decoding_path(encoded1, encoded2, encoded3, encoded4)
        
        final_output = self.fusion_output(decoded_output)
        
        return final_output