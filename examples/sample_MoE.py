# ==============================================================
# Example final Mixture of Experts Architecture
# Author: Elena González Prieto
# Date Modified: December 11th, 2025
# ==============================================================
import torch
from torch import nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class TaskSpecificMoE(nn.Module):
    """Separate MoE for classification and regression tasks"""
    def __init__(self, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.flatten = nn.Flatten()
        
        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(5, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        # Classification head
        self.class_moe = Expert(input_dim=256, hidden_dim=128,)
        self.class_head = nn.Linear(128, 4)
        
        # Regression experts
        self.reg_experts = nn.ModuleList([Expert(input_dim=256, hidden_dim=128) for _ in range(num_experts)])

        # Regression heads - one per expert
        self.reg_heads = nn.ModuleList([nn.Linear(128, 3) for _ in range(num_experts)])

    def forward(self, x):
        x = self.flatten(x)
        shared_features = self.shared_backbone(x)
        
        # Classification 
        class_features = self.class_moe(shared_features)
        class_out = self.class_head(class_features)
        
        # Compute gate from classification logits
        expert_indices = torch.argmax(class_out, dim=-1) 

        # Regression - only compute output from selected expert
        batch_size = shared_features.shape[0]
        reg_out = torch.zeros(batch_size, self.reg_heads[0].out_features, device=shared_features.device, dtype=shared_features.dtype)
    
        # Process each expert's samples
        for expert_idx in range(self.num_experts):
            # Find which samples use this expert
            mask = (expert_indices == expert_idx)
            if mask.any():
                # Only process samples assigned to this expert
                expert_features = self.reg_experts[expert_idx](shared_features[mask])
                expert_out = self.reg_heads[expert_idx](expert_features)
                reg_out[mask] = expert_out

        reg_out_fractions = F.softmax(reg_out, dim=-1)

        return class_out, reg_out_fractions