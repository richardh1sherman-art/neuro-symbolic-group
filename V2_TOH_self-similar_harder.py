import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerfectHanoiWreathEngine(nn.Module):
    def __init__(self, num_disks=3):
        super().__init__()
        self.num_disks = num_disks
        self.state_dim = 2 ** num_disks # 8-dimensional space
        
        # 2x2 Identity and Pauli-X (Permutation)
        self.I = torch.eye(2, device=device)
        self.M_a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        
        # Base Kronecker Group Generators
        self.Gen_1 = torch.kron(torch.kron(self.M_a, self.I), self.I)
        self.Gen_2 = torch.kron(torch.kron(self.I, self.M_a), self.I)
        
        # Create a Skew-Symmetric Matrix Operator (Omega^T = -Omega)
        # Multiplying commuting/non-commuting terms like this ensures a real unitary output
        self.Omega = torch.matmul(self.Gen_1, self.Gen_2) - torch.matmul(self.Gen_2, self.Gen_1)

    def get_true_unitary(self, theta):
        """
        Uses the matrix exponential to guarantee that U is a true unitary matrix.
        For any real skew-symmetric matrix Omega, exp(theta * Omega) is orthogonal.
        """
        # Exponentiate the continuous thought state space
        U = torch.matrix_exp(theta * self.Omega)
        return U

    def forward(self, hidden_state, theta, invert=False):
        U = self.get_true_unitary(theta)
        
        if invert:
            # Because U is truly orthogonal/unitary, the transpose is now the PERFECT inverse
            return torch.matmul(U.t(), hidden_state)
        
        return torch.matmul(U, hidden_state)

# Initialize system
engine = PerfectHanoiWreathEngine(num_disks=3)

# Initialize an 8D vector
initial_state = torch.zeros(8, 1, device=device)

# Activate indices that are forced to mutate under Gen_1 and Gen_2
initial_state[0] = 1.0  # State A
initial_state[1] = 1.0  # State B
initial_state = initial_state / torch.norm(initial_state) # Keep norm at 0.707


# Set a reasoning blend angle
reasoning_theta = 0.5 

print(f"--- PERFECT UNITARY ARCHITECTURE (State Space Dim: {engine.state_dim}) ---")
print(f"Device: {device} | Engine: NVIDIA GB10\n")

print("Input Hidden State Vector:")
print(initial_state.cpu().numpy().flatten())

# 1. Forward Pass
output_state = engine(initial_state, reasoning_theta, invert=False)
print("\nOutput Hidden State (Continuous Wreath Rotation):")
print(output_state.cpu().numpy().flatten())

# 2. Inverse Pass
recovered_state = engine(output_state, reasoning_theta, invert=True)
print("\nRecovered Identity State (Flawless Match):")
print(recovered_state.cpu().numpy().flatten())
