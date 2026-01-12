import math

import torch
import torch.nn as nn

# ============================================================================
# UTILS
# ============================================================================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, D) -> weights: (B, T, 1)
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


# ============================================================================
# KEYPOINT MODEL
# ============================================================================


class TransformerEncoderKeypoints(nn.Module):
    def __init__(self, num_classes, T=32, V=75, C=3, d_model=128, dropout_p=0.1):
        super().__init__()
        self.joint_embed = nn.Linear(C, 16)
        self.temporal_proj = nn.Linear(16 * V, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=T)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout_p,
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
        x = self.temporal_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.pool(x)
        return self.fc(x)


class BiLSTMKeypoints(nn.Module):
    """
    BiLSTM model for keypoint sequences.

    Input: (B, 3, T, 75)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes, T=32, V=75, C=3, hidden_dim=128, dropout_p=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=C * V,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.attn = TemporalAttentionPooling(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
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
    def __init__(self, num_classes, input_dim=1024, dropout_p=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout_p),  # Increased Dropout
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # Increased Dropout
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class RGBFlowClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=1024, dropout_p=0.1):
        super().__init__()

        self.rgb_branch = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout_p),
            nn.Linear(input_dim, 128),  # Drastically reduced from 256
            nn.ReLU(),
        )
        self.flow_branch = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout_p),
            nn.Linear(input_dim, 128),  # Drastically reduced from 256
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 2, 1024) -> index 0 is RGB, 1 is Flow
        rgb = self.rgb_branch(x[:, 0])
        flow = self.flow_branch(x[:, 1])
        fused = torch.cat([rgb, flow], dim=-1)
        return self.fusion(fused)


# ============================================================================
# TRIPLE STREAM FUSION (KPS + RGB + FLOW)
# ============================================================================


class TripleStreamFusion(nn.Module):
    def __init__(
        self, num_classes, T=32, kps_d_model=128, feature_dim=1024, dropout_p=0.1
    ):
        super().__init__()

        # --- KPS Branch (The Strongest Signal) ---
        self.kps_embed = nn.Linear(3, 16)
        self.kps_temporal_proj = nn.Linear(16 * 75, kps_d_model)
        self.pos_enc = PositionalEncoding(kps_d_model, max_len=T)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=kps_d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout_p,
            batch_first=True,
        )
        self.kps_encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.kps_pool = TemporalAttentionPooling(kps_d_model)
        self.kps_norm = nn.LayerNorm(kps_d_model)

        # --- RGB & Flow Branches (The "Bottleneck") ---
        # We crush 1024 dims down to 32 dims IMMEDIATELY.
        # This forces the model to throw away noise and keep only the best features.
        self.rgb_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_p),  # High dropout on raw features
            nn.Linear(feature_dim, 32),  # BOTTLENECK: 1024 -> 32
            nn.ReLU(),
        )
        self.flow_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_p),
            nn.Linear(feature_dim, 32),  # BOTTLENECK: 1024 -> 32
            nn.ReLU(),
        )

        # --- Fusion ---
        # 128 (KPS) + 32 (RGB) + 32 (Flow) = 192
        # The KPS signal now dominates the vector (128 vs 64), which is what we want.
        fusion_dim = kps_d_model + 32 + 32
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, kps, rgb, flow):
        # KPS Path
        B, C, T, V = kps.shape
        x_kps = kps.permute(0, 2, 3, 1)
        x_kps = self.kps_embed(x_kps).reshape(B, T, -1)
        x_kps = self.kps_temporal_proj(x_kps)
        x_kps = self.pos_enc(x_kps)
        x_kps = self.kps_encoder(x_kps)
        x_kps = self.kps_pool(x_kps)
        x_kps = self.kps_norm(x_kps)

        # Feature Path
        x_rgb = self.rgb_proj(rgb)
        x_flow = self.flow_proj(flow)

        concat = torch.cat([x_kps, x_rgb, x_flow], dim=1)
        return self.classifier(concat)


class KPSFlowFusion(nn.Module):
    def __init__(
        self, num_classes, T=32, kps_d_model=128, feature_dim=1024, dropout_p=0.1
    ):
        super().__init__()

        # --- Branch 1: Keypoints (The Star) ---
        self.kps_embed = nn.Linear(3, 16)
        self.kps_temporal_proj = nn.Linear(16 * 75, kps_d_model)
        self.pos_enc = PositionalEncoding(kps_d_model, max_len=T)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=kps_d_model,
            nhead=2,
            dim_feedforward=256,
            dropout=dropout_p,
            batch_first=True,
        )
        self.kps_encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.kps_pool = TemporalAttentionPooling(kps_d_model)
        self.kps_norm = nn.LayerNorm(kps_d_model)

        # --- Branch 2: Flow (The Strong Support) ---
        # Flow features are 1024 dim. We project them to match KPS size (128).
        self.flow_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_p),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
        )

        # --- Fusion ---
        # Concatenate 128 (KPS) + 128 (Flow) = 256
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, kps, flow):
        # KPS Path
        B, C, T, V = kps.shape
        x_kps = kps.permute(0, 2, 3, 1)
        x_kps = self.kps_embed(x_kps).reshape(B, T, -1)
        x_kps = self.kps_temporal_proj(x_kps)
        x_kps = self.pos_enc(x_kps)
        x_kps = self.kps_encoder(x_kps)
        x_kps = self.kps_pool(x_kps)
        x_kps = self.kps_norm(x_kps)

        # Flow Path
        x_flow = self.flow_proj(flow)

        # Fuse
        concat = torch.cat([x_kps, x_flow], dim=1)
        return self.classifier(concat)
