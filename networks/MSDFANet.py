import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from functools import partial

tensor_rotate = partial(rotate, interpolation=InterpolationMode.BILINEAR)


# ==================== 1. BASIC BUILDING BLOCKS ====================
class ResidualBlock(nn.Module):
    """Basic residual block with skip connection for gradient flow"""
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return F.relu(out)


class FeatureConnector(nn.Module):
    """Multi-scale feature connector with residual connection"""
    def __init__(self, in_ch, out_ch, scale_factor=0.5):
        super(FeatureConnector, self).__init__()
        self.downsample = partial(F.interpolate, scale_factor=scale_factor, 
                                 mode='area', recompute_scale_factor=True)
        
        if not in_ch == out_ch:
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch))
        else:
            shortcut = None
        self.connect_conv = ResidualBlock(in_ch, out_ch, shortcut=shortcut)

    def forward(self, x):
        x = self.downsample(x)
        x = self.connect_conv(x)
        return x


# ==================== 2. CORE FEATURE EXTRACTION MODULES ====================
class MultiScaleDirectionalInitBlock(nn.Module):
    """Multi-scale directional feature initialization with rotational convolutions"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, 
                 dilation=1, bias=False, strip=9):
        super(MultiScaleDirectionalInitBlock, self).__init__()
        
        # Base convolutional path
        self.conv_base = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size, stride, padding=padding, 
                     dilation=dilation, bias=bias),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
        )

        # Multi-directional convolutional paths
        self.directional_conv1 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride, padding=(0, strip//2))
        self.directional_conv2 = nn.Conv2d(in_ch, 8, (strip, 1), stride=stride, padding=(strip//2, 0))
        self.directional_conv3 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride, padding=(0, strip//2))
        self.directional_conv4 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride, padding=(0, strip//2))

        # Channel refinement module
        self.channel_refinement = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, padding=0, dilation=dilation, bias=bias),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
        )
        self.rotation_angles = [0, 45, 90, 135, 180]

    def forward(self, x):
        # Extract features from different directional paths
        x1 = self.directional_conv1(x)  # Horizontal
        x2 = self.directional_conv2(x)  # Vertical
        x3 = self.conv_base(x)          # Base features
        x4 = self.directional_conv3(tensor_rotate(x, self.rotation_angles[1]))  # 45° rotated
        x5 = self.directional_conv4(tensor_rotate(x, self.rotation_angles[3]))  # 135° rotated

        # Combine directional features
        directional_features = torch.cat((x1, x2, 
                         tensor_rotate(x4, -self.rotation_angles[1]),
                         tensor_rotate(x5, -self.rotation_angles[3])), 1)
        
        # Final feature integration
        out = torch.cat((self.channel_refinement(directional_features), x3), 1)
        return out


class SpatialTransformInitBlock(nn.Module):
    """Spatial transformation-based feature initialization with H/V transforms"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, 
                 dilation=1, bias=False, strip=9):
        super(SpatialTransformInitBlock, self).__init__()
        
        self.conv_base = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size, stride, padding=padding, 
                     dilation=dilation, bias=bias),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
        )

        self.directional_conv1 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride, padding=(0, strip//2))
        self.directional_conv2 = nn.Conv2d(in_ch, 8, (strip, 1), stride=stride, padding=(strip//2, 0))
        self.directional_conv3 = nn.Conv2d(in_ch, 8, (9, 1), stride=stride, padding=(4, 0))
        self.directional_conv4 = nn.Conv2d(in_ch, 8, (1, 9), stride=stride, padding=(0, 4))

        self.channel_refinement = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, padding=0, dilation=dilation, bias=bias),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x1 = self.directional_conv1(x)
        x2 = self.directional_conv2(x)
        x3 = self.conv_base(x)
        x4 = self.directional_conv3(self.horizontal_transform(x, 2))
        x5 = self.directional_conv4(self.vertical_transform(x, 2))

        directional_features = torch.cat((x1, x2, 
                         self.inverse_horizontal_transform(x4),
                         self.inverse_vertical_transform(x5)), 1)
        
        out = torch.cat((self.channel_refinement(directional_features), x3), 1)
        return out

    def horizontal_transform(self, x, stride=1):
        """Apply horizontal spatial transformation"""
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]*stride]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - stride)
        return x

    def inverse_horizontal_transform(self, x):
        """Inverse horizontal spatial transformation"""
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def vertical_transform(self, x, stride=1):
        """Apply vertical spatial transformation"""
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]*stride]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - stride)
        return x.permute(0, 1, 3, 2)

    def inverse_vertical_transform(self, x):
        """Inverse vertical spatial transformation"""
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


# ==================== 3. ENCODER ARCHITECTURE ====================
class HierarchicalEncoderStage(nn.Module):
    """Hierarchical encoder stage with multi-scale feature integration"""
    def __init__(self, scale_factor, in_c=64, res_layers=64, num_res_blocks=3, stride=1):
        super(HierarchicalEncoderStage, self).__init__()
        self.residual_path = self._build_residual_path(in_c, res_layers, num_res_blocks, stride)
        self.skip_connection = FeatureConnector(3, res_layers, scale_factor=scale_factor)

    def forward(self, x_input, res_input):
        out1 = self.residual_path(res_input)  # Residual path features
        out2 = self.skip_connection(x_input)   # Skip connection features
        out = torch.cat((out1, out2), 1)     # Feature concatenation
        return out

    def _build_residual_path(self, in_ch, out_ch, block_num, stride=1):
        """Build residual path with shortcut connection"""
        shortcut = None
        if not in_ch == out_ch or not stride == 1:
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)


# ==================== 4. DECODER ARCHITECTURE ====================
class MultiDirectionalDecoder(nn.Module):
    """Multi-directional decoder with feature refinement and upsampling"""
    def __init__(self, in_channels, n_filters, BatchNorm=nn.BatchNorm2d, use_upsampling=True, strip=9):
        super(MultiDirectionalDecoder, self).__init__()
        out_pad = 1 if use_upsampling else 0
        stride = 2 if use_upsampling else 1

        # Feature compression paths
        self.feature_compression1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )

        self.feature_compression2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            BatchNorm(in_channels // 2),
            nn.ReLU(inplace=True), 
        )

        # Directional convolution paths
        self.directional_deconv1 = nn.Conv2d(in_channels // 4, in_channels // 4, (1, strip), padding=(0, strip//2))
        self.directional_deconv2 = nn.Conv2d(in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0))
        self.directional_deconv3 = nn.Conv2d(in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0))
        self.directional_deconv4 = nn.Conv2d(in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0))

        # Feature refinement modules
        self.feature_refinement1 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )
        self.feature_refinement2 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )
        self.feature_refinement3 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )
        self.feature_refinement4 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )

        # Upsampling path
        self.upsampling_path = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 4 + in_channels // 4,
                             3, stride=stride, padding=1, output_padding=out_pad),
            nn.BatchNorm2d(in_channels // 4 + in_channels // 4),
            nn.ReLU(inplace=True), 
        )

        # Output projection
        self.output_projection = nn.Conv2d(in_channels // 4 + in_channels // 4, n_filters, 1)
        self.output_norm = BatchNorm(n_filters)
        self.output_activation = nn.ReLU()

    def forward(self, x, inp=False):
        # Feature compression
        compressed1 = self.feature_compression1(x)
        compressed2 = self.feature_compression2(x)

        # Multi-directional feature extraction
        dir1 = self.directional_deconv1(compressed1)
        dir2 = self.directional_deconv2(compressed1)
        dir3 = tensor_rotate(self.directional_deconv3(tensor_rotate(compressed1, 45)), -45)
        dir4 = tensor_rotate(self.directional_deconv4(tensor_rotate(compressed1, 135)), -135)

        # Feature refinement with skip connections
        refined1 = self.feature_refinement1(torch.cat((dir1, compressed2), 1))
        refined2 = self.feature_refinement2(torch.cat((dir2, compressed2), 1))
        refined3 = self.feature_refinement3(torch.cat((dir3, compressed2), 1))
        refined4 = self.feature_refinement4(torch.cat((dir4, compressed2), 1))
        
        # Feature aggregation
        aggregated_features = torch.cat((refined1, refined2, refined3, refined4), 1)
        upsampled_features = self.upsampling_path(aggregated_features)
        
        # Final projection
        output = self.output_projection(upsampled_features)
        output = self.output_norm(output)
        output = self.output_activation(output)
        return output


class SpatialTransformDecoder(nn.Module):
    """Spatial transform decoder with H/V transformation operations"""
    def __init__(self, in_channels, n_filters, BatchNorm=nn.BatchNorm2d, use_upsampling=True, strip=9):
        super(SpatialTransformDecoder, self).__init__()
        out_pad = 1 if use_upsampling else 0
        stride = 2 if use_upsampling else 1

        self.feature_compression1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )

        self.feature_compression2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            BatchNorm(in_channels // 2),
            nn.ReLU(inplace=True), 
        )

        self.directional_deconv1 = nn.Conv2d(in_channels // 4, in_channels // 4, (1, strip), padding=(0, strip//2))
        self.directional_deconv2 = nn.Conv2d(in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0))
        self.directional_deconv3 = nn.Conv2d(in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0))
        self.directional_deconv4 = nn.Conv2d(in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4))

        self.feature_refinement1 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )
        self.feature_refinement2 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )
        self.feature_refinement3 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )
        self.feature_refinement4 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), 
        )

        self.upsampling_path = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 4 + in_channels // 4,
                             3, stride=stride, padding=1, output_padding=out_pad),
            nn.BatchNorm2d(in_channels // 4 + in_channels // 4),
            nn.ReLU(inplace=True), 
        )

        self.output_projection = nn.Conv2d(in_channels // 4 + in_channels // 4, n_filters, 1)
        self.output_norm = BatchNorm(n_filters)
        self.output_activation = nn.ReLU()

    def forward(self, x, inp=False):
        compressed1 = self.feature_compression1(x)
        compressed2 = self.feature_compression2(x)

        dir1 = self.directional_deconv1(compressed1)
        dir2 = self.directional_deconv2(compressed1)
        dir3 = self.inverse_horizontal_transform(self.directional_deconv3(self.horizontal_transform(compressed1)))
        dir4 = self.inverse_vertical_transform(self.directional_deconv4(self.vertical_transform(compressed1)))

        refined1 = self.feature_refinement1(torch.cat((dir1, compressed2), 1))
        refined2 = self.feature_refinement2(torch.cat((dir2, compressed2), 1))
        refined3 = self.feature_refinement3(torch.cat((dir3, compressed2), 1))
        refined4 = self.feature_refinement4(torch.cat((dir4, compressed2), 1))
        
        aggregated_features = torch.cat((refined1, refined2, refined3, refined4), 1)
        upsampled_features = self.upsampling_path(aggregated_features)
        
        output = self.output_projection(upsampled_features)
        output = self.output_norm(output)
        output = self.output_activation(output)
        return output

    def horizontal_transform(self, x):
        """Horizontal spatial transformation"""
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inverse_horizontal_transform(self, x):
        """Inverse horizontal transformation"""
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def vertical_transform(self, x):
        """Vertical spatial transformation"""
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inverse_vertical_transform(self, x):
        """Inverse vertical transformation"""
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


# ==================== 5. ATTENTION AND ENHANCEMENT MODULES ====================
class DepthwiseSeparableConvolution(nn.Module):
    """Depthwise separable convolution for efficient feature extraction"""
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_c, in_c, kernel, stride,
            padding=padding, dilation=dilation, groups=in_c
        )
        self.pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MultiScaleChannelAttention(nn.Module):
    """Multi-scale channel attention with parallel convolution paths"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(channel, channel//reduction, 1),      # 1x1 convolution
            nn.Conv2d(channel, channel//reduction, 3, padding=1),      # 3x3 convolution
            nn.Conv2d(channel, channel//reduction, 3, padding=3, dilation=3),  # Dilated 3x3
            nn.Conv2d(channel, channel//reduction, 5, padding=5, dilation=5)  # Dilated 5x5
        ])
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(4*(channel//reduction), channel, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract multi-scale features
        multi_scale_features = [F.interpolate(conv(x), size=x.shape[2:], mode='bilinear') 
                              for conv in self.multi_scale_convs]
        # Fuse features and apply attention
        attention_weights = self.feature_fusion(torch.cat(multi_scale_features, dim=1))
        return x * attention_weights


class MultiScaleFeatureEnhancer(nn.Module):
    """Multi-scale feature enhancement with parallel processing paths"""
    def __init__(self, channel):
        super().__init__()
        self.parallel_branches = nn.ModuleList([
            # Average pooling branch
            nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                nn.Conv2d(channel, channel//4, 1)
            ),
            # Standard convolution branch
            nn.Conv2d(channel, channel//4, 3, padding=1),
            # Dilated convolution branch 1
            nn.Conv2d(channel, channel//4, 3, padding=3, dilation=3),
            # Dilated convolution branch 2
            nn.Conv2d(channel, channel//4, 3, padding=5, dilation=5)
        ])
        self.feature_fusion = nn.Conv2d(channel, channel, 1)
        
    def forward(self, x):
        # Process through parallel branches
        branch_outputs = [branch(x) for branch in self.parallel_branches]
        # Fuse features with residual connection
        enhanced_features = self.feature_fusion(torch.cat(branch_outputs, dim=1))
        return x + enhanced_features  # Residual connection


class ConfidenceAwareRefinement(nn.Module):
    """Confidence-aware output refinement with uncertainty estimation"""
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.confidence_estimator = nn.Sequential(
            DepthwiseSeparableConvolution(in_channels, 32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Generate confidence weights for output refinement
        
        Args:
            features: Input features for confidence estimation
            
        Returns:
            Confidence weights for adaptive refinement
        """
        return self.confidence_estimator(features)


# ==================== 6. MAIN NETWORK ARCHITECTURE ====================
class MSDFANet(nn.Module):
    """
    MSDFA-Net: Multi-Scale Directional Feature Attention Network
    A deep learning architecture for high-precision road extraction from remote sensing imagery
    
    Key Innovations:
    - Multi-scale directional feature extraction
    - Hierarchical encoder-decoder architecture  
    - Multi-scale channel attention mechanism
    - Spatial transformation operations
    - Confidence-aware output refinement
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        feature_dims = [64, 128, 256, 512]
        
        # Multi-scale directional feature initialization
        self.feature_initializer = MultiScaleDirectionalInitBlock(in_channels, 64, stride=2) 
        
        # Hierarchical encoder stages
        self.encoder_stage1 = HierarchicalEncoderStage(0.5, feature_dims[0], feature_dims[0], 3, stride=1)
        self.encoder_stage2 = HierarchicalEncoderStage(0.25, feature_dims[1], feature_dims[1], 4, stride=2)
        self.encoder_stage3 = HierarchicalEncoderStage(0.125, feature_dims[2], feature_dims[2], 6, stride=2)
        self.encoder_stage4 = HierarchicalEncoderStage(0.0625, feature_dims[3], feature_dims[3], 3, stride=2)

        # Feature integration connectors
        self.feature_integrator5 = FeatureConnector(1024, 512, scale_factor=1)
        self.feature_integrator4 = FeatureConnector(512, 256, scale_factor=1)
        self.feature_integrator3 = FeatureConnector(256, 128, scale_factor=1)
        self.feature_integrator2 = FeatureConnector(128, 64, scale_factor=1)
        
        # Multi-directional decoder stages
        self.decoder_stage4 = MultiDirectionalDecoder(feature_dims[3], feature_dims[2], use_upsampling=True)
        self.decoder_stage3 = MultiDirectionalDecoder(feature_dims[2], feature_dims[1], use_upsampling=True)
        self.decoder_stage2 = MultiDirectionalDecoder(feature_dims[1], feature_dims[0], use_upsampling=True)
        self.decoder_stage1 = MultiDirectionalDecoder(feature_dims[0], feature_dims[0], use_upsampling=True)
        
        # Feature enhancement modules
        self.channel_attention = MultiScaleChannelAttention(64)
        self.feature_enhancer = MultiScaleFeatureEnhancer(64)
        self.confidence_refinement = ConfidenceAwareRefinement(64, num_classes)
        
        # Final output projection
        self.output_projector = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 3, padding=1)
        )

    def forward(self, x):
        """
        Forward pass through the MSDFA-Net architecture
        
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tensor: Output segmentation map of shape (B, 1, H, W)
        """
        # Feature initialization
        initial_features = self.feature_initializer(x)
        
        # Hierarchical encoding
        encoded1 = self.encoder_stage1(x, initial_features)
        encoded2 = self.encoder_stage2(x, encoded1)
        encoded3 = self.encoder_stage3(x, encoded2)
        encoded4 = self.encoder_stage4(x, encoded3)
        
        # Feature integration and decoding
        integrated4 = self.feature_integrator5(encoded4)
        decoded4 = self.decoder_stage4(integrated4) + self.feature_integrator4(encoded3)
        decoded3 = self.decoder_stage3(decoded4) + self.feature_integrator3(encoded2)
        decoded2 = self.decoder_stage2(decoded3) + self.feature_integrator2(encoded1)
        decoded1 = self.decoder_stage1(decoded2)
        
        # Feature enhancement
        enhanced_features = self.feature_enhancer(decoded1)
        attended_features = self.channel_attention(enhanced_features)
        
        # Confidence-aware refinement
        base_output = self.output_projector(attended_features)
        confidence_weights = self.confidence_refinement(attended_features)
        final_output = base_output * confidence_weights
        
        return final_output