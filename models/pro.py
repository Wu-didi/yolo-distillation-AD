import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class ProbabilisticAttention(nn.Module):
    def __init__(self, d_model, num_samples=5):
        super(ProbabilisticAttention, self).__init__()
        self.query_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.key_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.value_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        
        # Convolutional layers for mean and std deviation
        self.mean_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.std_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Scaled dot-product attention factor
        self.scale = d_model ** 0.5
        
        # Number of samples for averaging
        self.num_samples = num_samples

    def forward(self, queries, keys, values):
        # Convolutional projections
        q = self.query_conv(queries)
        k = self.key_conv(keys)
        v = self.value_conv(values)

        # Calculate mean and standard deviation
        mean = self.mean_conv(q - k)
        std = F.softplus(self.std_conv(q - k))  # Ensure non-negative std deviation
        
        # Sample attention weights from Gaussian distribution and average
        attention_weights = 0
        for _ in range(self.num_samples):
            eps = torch.randn_like(std)
            sampled_attention_weights = mean + eps * std
            attention_weights += sampled_attention_weights / self.num_samples

        # Reshape for matrix multiplication
        bs, c, h, w = attention_weights.shape
        attention_weights = attention_weights.view(bs, c, -1)
        k = k.view(bs, c, -1)
        v = v.view(bs, c, -1)

        # Scaled dot-product attention
        attention_scores = torch.matmul(attention_weights.permute(0, 2, 1), k) / self.scale
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Calculate attention output
        attention_output = torch.matmul(attention_scores, v.permute(0, 2, 1))
        attention_output = attention_output.permute(0, 2, 1).view(bs, c, h, w)
        
        # Apply layer normalization
        attention_output = self.layer_norm(attention_output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return attention_output


if __name__ == '__main__':
    # Test probabilistic attention
    d_model = 512
    queries = torch.randn(4, d_model, 80, 80)
    keys = torch.randn(4, d_model, 80, 80)
    values = torch.randn(4, d_model, 80, 80)

    # Create probabilistic attention layer
    probabilistic_attention = ProbabilisticAttention(d_model)

    # Forward pass
    output = probabilistic_attention(queries, keys, values)
    print(output.shape)  # Output shape: (4, 10, 512)