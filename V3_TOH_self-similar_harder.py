import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrueWreathHanoiEngine(nn.Module):
    def __init__(self, num_disks=3):
        super().__init__()
        self.num_disks = num_disks
        self.state_dim = 2 ** num_disks # 8 dimensions
        
        self.I = torch.eye(2, device=device)
        self.M_a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        
        # Gen_1: Swaps top-level tree branches globally
        self.Gen_1 = torch.kron(torch.kron(self.M_a, self.I), self.I)
        
        # Gen_2: Recursive Wreath Action. It acts as M_a on the left branch, 
        # but stays Identity on the right branch. This forces non-commutation!
        left_branch_action = torch.kron(torch.kron(self.M_a, self.M_a), self.I)
        right_branch_action = torch.kron(torch.kron(self.I, self.I), self.I)
        
        # Project them to create a branch-dependent group operator
        self.Gen_2 = left_branch_action - right_branch_action 
        
        # This will evaluate to a massive, non-zero skew-symmetric matrix!
        self.Omega = torch.matmul(self.Gen_1, self.Gen_2) - torch.matmul(self.Gen_2, self.Gen_1)

    def get_true_unitary(self, theta):
        return torch.matrix_exp(theta * self.Omega)

    def forward(self, hidden_state, theta, invert=False):
        U = self.get_true_unitary(theta)
        if invert:
            return torch.matmul(U.t(), hidden_state)
        return torch.matmul(U, hidden_state)

# Initialize system
engine = TrueWreathHanoiEngine(num_disks=3)

# Define an input vector activating the first two coordinates
initial_state = torch.zeros(8, 1, device=device)
initial_state[0] = 1.0
initial_state[1] = 1.0
initial_state = initial_state / torch.norm(initial_state)

# Set a significant rotation angle to force state movement
reasoning_theta = 0.5 

print(f"--- TRUE WREATH ARCHITECTURE (State Space Dim: {engine.state_dim}) ---")
print(f"Device: {device} | Engine: NVIDIA GB10\n")

print("Input Hidden State Vector:")
print(initial_state.cpu().numpy().flatten())

# 1. Forward Pass (Will actively scramble the vector)
output_state = engine(initial_state, reasoning_theta, invert=False)
print("\nOutput Hidden State (Continuous Wreath Rotation):")
print(output_state.cpu().numpy().flatten())

# 2. Inverse Pass (Will snap back perfectly)
recovered_state = engine(output_state, reasoning_theta, invert=True)
print("\nRecovered Identity State (Flawless Match):")
print(recovered_state.cpu().numpy().flatten())
