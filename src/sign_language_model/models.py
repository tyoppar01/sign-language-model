import torch
import torch.nn as nn
import math


# ============================================================================
# KEYPOINT MODELS
# ============================================================================


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    Input: (B, T, D), where B=batch size, T=temporal length, D=feature dimension
    Output: (B, T, D)
    """

    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalAttentionPooling(nn.Module):
    """
    Temporal attention pooling layer.
    Input: (B, T, D), where B=batch size, T=temporal length, D=feature dimension
    Output: (B, D)
    """

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, D)
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class TransformerEncoderKeypoints(nn.Module):
    """
    Transformer Encoder model for keypoint sequences.

    Input: (B, 3, T, 75)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes, T=32, V=75, C=3, d_model=128):
        super().__init__()

        self.joint_embed = nn.Linear(C, 16)
        self.temporal_proj = nn.Linear(16 * V, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=32)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)

        self.pool = TemporalAttentionPooling(d_model)

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, T, V, C)
        x = self.joint_embed(x)  # (B, T, V, 16)
        x = x.reshape(B, T, -1)  # (B, T, V*16)
        x = self.temporal_proj(x)  # (B, T, d_model)

        x = self.pos_enc(x)  # (B, T, d_model)
        x = self.encoder(x)  # (B, T, d_model)
        x = self.pool(x)  # (B, d_model)
        return self.fc(x)


class BiLSTMKeypoints(nn.Module):
    """
    BiLSTM model for keypoint sequences.

    Input: (B, 3, T, 75)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes, T=32, V=75, C=3, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=C * V,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.attn = TemporalAttentionPooling(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V)  # (B, T, C*V)
        x, _ = self.lstm(x)  # (B, T, hidden_dim*2)
        x = self.attn(x)  # (B, hidden_dim*2)
        return self.fc(x)


# ============================================================================
# RGB/FLOW MODELS
# ============================================================================
class I3DClassifier(nn.Module):
    """
    Simple classifier for I3D-extracted RGB/Flow features.
    Input: (B, 2, 1024)  # RGB and Flow features
    Output: (B, num_classes)
    """

    def __init__(self, num_classes, input_dim=1024):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, 1024)
        return self.net(x)


# ============================================================================
# MULTI-MODAL MODELS
# ============================================================================


class RGBFlowClassifier(nn.Module):
    """
    Late fusion of RGB and Flow features.
    Input: (B, 2, 1024)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes, input_dim=1024):
        super().__init__()

        self.rgb_branch = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
        )

        self.flow_branch = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, 2, 1024)
        rgb = self.rgb_branch(x[:, 0])
        flow = self.flow_branch(x[:, 1])
        fused = torch.cat([rgb, flow], dim=-1)
        return self.fusion(fused)


class LateFusionClassifier(nn.Module):
    """
    Late fusion of keypoint and RGB+Flow logits with learnable weights.
    Input: logits from each modality (B, num_classes)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes, kps_model: nn.Module, rgb_flow_model: nn.Module):
        super().__init__()

        # Freeze sub-models
        self.kps_model = kps_model
        self.rgb_flow_model = rgb_flow_model

        self.weights = nn.Parameter(torch.ones(2))  # kps, rgb+flow

    def forward(self, x_kps, x_rgb_flow):
        # x_kps: (B, 3, T, 75)
        # x_rgb_flow: (B, 2, 1024)
        logits_kps = self.kps_model(x_kps)
        logits_rgb_flow = self.rgb_flow_model(x_rgb_flow)

        w = torch.softmax(self.weights, dim=0)
        return w[0] * logits_kps + w[1] * logits_rgb_flow
