import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DefinitiveWreathEngine(nn.Module):
    def __init__(self, num_disks=3):
        super().__init__()
        self.num_disks = num_disks
        self.state_dim = 2 ** num_disks # 8 dimensions
        
        self.I = torch.eye(2, device=device)
        self.M_a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        
        # Gen_1: A global swap across the entire 8D space
        self.Gen_1 = torch.kron(torch.kron(self.M_a, self.I), self.I)
        
        # Gen_2: A TRUE conditional branch action using block diagonals.
        # Top 4x4 block (Left branch) executes a nested swap.
        # Bottom 4x4 block (Right branch) is strictly the identity.
        left_action = torch.kron(self.M_a, self.M_a)
        right_action = torch.eye(4, device=device)
        
        # Physically stack them as separate branches of the tree
        self.Gen_2 = torch.block_diag(left_action, right_action)
        
        # This is mathematically guaranteed to be a non-zero, active commutator!
        self.Omega = torch.matmul(self.Gen_1, self.Gen_2) - torch.matmul(self.Gen_2, self.Gen_1)

    def get_true_unitary(self, theta):
        return torch.matrix_exp(theta * self.Omega)

    def forward(self, hidden_state, theta, invert=False):
        U = self.get_true_unitary(theta)
        if invert:
            return torch.matmul(U.t(), hidden_state)
        return torch.matmul(U, hidden_state)

# Initialize system
engine = DefinitiveWreathEngine(num_disks=3)

# Define an input vector activating the first two coordinates
initial_state = torch.zeros(8, 1, device=device)
initial_state[0] = 1.0
initial_state[1] = 1.0
initial_state = initial_state / torch.norm(initial_state)

# Set a significant rotation angle to force a visible shift
reasoning_theta = 0.5 

print(f"--- DEFINITIVE WREATH ARCHITECTURE (State Space Dim: {engine.state_dim}) ---")
print(f"Device: {device} | Engine: NVIDIA GB10\n")

print("Input Hidden State Vector:")
print(initial_state.cpu().numpy().flatten())

# 1. Forward Pass (Will actively scramble the vector)
output_state = engine(initial_state, reasoning_theta, invert=False)
print("\nOutput Hidden State (Continuous Wreath Rotation):")
print(output_state.cpu().numpy().flatten())

# 2. Inverse Pass (Will snap back perfectly to input)
recovered_state = engine(output_state, reasoning_theta, invert=True)
print("\nRecovered Identity State (Flawless Match):")
print(recovered_state.cpu().numpy().flatten())
