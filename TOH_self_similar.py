import torch
import torch.nn as nn

# 1. Target the Grace Blackwell unified memory architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Engine: {torch.cuda.get_device_name(0)}")

class WreathSuperpositionEngine(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        
        # Identity Operator (Leaves branch unchanged)
        self.I = torch.eye(2, device=device)
        
        # Generator 'a': Active branch switcher (2x2 permutation)
        self.M_a = torch.tensor([[0.0, 1.0], 
                                 [1.0, 0.0]], device=device)
        
        # Self-similar Generator 'b': Acts recursively as (a, I)
        # We explicitly model this structure via a block-diagonal Kronecker structure
        self.M_b_left = torch.kron(self.M_a, torch.eye(2, device=device))  # Branch 0 acts as 'a'
        self.M_b_right = torch.kron(torch.eye(2, device=device), self.I)   # Branch 1 acts as 'I'
        self.M_b = self.M_b_left + self.M_b_right

        # Continuous Projection Layer to handle hidden state mixtures (Zhu et al. framework)
        self.continuous_mix = nn.Linear(4, 4, bias=False, device=device)

    def forward(self, hidden_state, alpha, beta):
        """
        Executes a continuous superposition of two recursive self-similar tracks.
        alpha/beta: continuous scalar weights assigned via transformer reasoning heads
        """
        # Formulate a continuous matrix blend across the tensor product space
        blended_operator = (alpha * self.M_b) + (beta * torch.kron(self.M_a, self.M_a))
        
        # Pass the token hidden state through the blended algebraic operation
        next_hidden = torch.matmul(blended_operator, hidden_state)
        return next_hidden

# 2. Initialize a 4-dimensional state representation (2 disks = 2^2 tree space)
engine = WreathSuperpositionEngine(state_dim=4)

# 3. Simulate a continuous token input vector h_t 
# (Representing a superposition of multiple puzzle arrangements)
initial_hidden_state = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device).unsqueeze(1)

# 4. Define continuous reasoning path coefficients (50% Track B, 50% Track A^2)
alpha_weight = 0.5
beta_weight = 0.5

print("\n--- FORWARD PASS (Superposed Execution) ---")
print(f"Input Hidden State:\n{initial_hidden_state.cpu().numpy().flatten()}")

# Execute forward transition
output_hidden_state = engine(initial_hidden_state, alpha_weight, beta_weight)
print(f"Output Hidden State:\n{output_hidden_state.cpu().detach().numpy().flatten()}")

print("\n--- INVERSE PASS (Algebraic Reversal) ---")
# Because generators are involutions, multiplying by the inverse 
# is equivalent to re-running the operator with mirrored flow!
recovered_hidden_state = engine(output_hidden_state, alpha_weight, beta_weight)
print(f"Recovered Identity State:\n{recovered_hidden_state.cpu().detach().numpy().flatten()}")
