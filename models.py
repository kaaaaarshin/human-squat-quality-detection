import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    V8 Specific Temporal Attention: Abandons "all frames equal" compression natively.
    Forces the network to explicitly weight crucial anomaly inflection slices explicitly.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim/2), 1)
        )

    def forward(self, lstm_out):
        # lstm_out shape: [batch, T, hidden_dim]
        attn_scores = self.attention(lstm_out) # [batch, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1) 
        context = torch.sum(attn_weights * lstm_out, dim=1) # [batch, hidden_dim]
        return context, attn_weights

class AttentionLSTMPredictor(nn.Module):
    """
    V8 Elite Model Matrix: Temporal Attention + Contrastive Latent Extractor
    Maps fake MediaPipe geometry securely into pure Fit3D Latent representations identically.
    """
    def __init__(self, input_dim=21, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(hidden_dim * 2)
        
        self.latent_proj = nn.Linear(hidden_dim * 2, 64) 
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3) # [depth, valgus, back]
        )

    def forward(self, x, return_latent=False):
        lstm_out, _ = self.encoder(x)
        context, attn = self.attention(lstm_out)
        
        latent = self.latent_proj(context)
        preds = self.fc(latent)
        
        if return_latent:
            return preds, latent
        return preds
