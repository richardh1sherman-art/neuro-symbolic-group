import torch
import torch.nn as nn
import torch.optim as optim
import time
from z3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftPolicyNetwork(nn.Module):
    """A soft initialized policy network designed to break logit plateaus"""
    def __init__(self, steps=2):
        super().__init__()
        # Soft random values centered around 0.0 to keep gradients highly active
        self.policy_logits = nn.Parameter(torch.randn(steps, 2, device=device) * 0.1, requires_grad=True)

    def forward(self):
        return torch.softmax(self.policy_logits, dim=-1)

def run_z3_oracle(num_disks, steps, start_idx, target_idx):
    """The hard symbolic filter. Returns the exact discrete operator index sequence."""
    s = Solver()
    states = [BitVec(f"state_{t}", num_disks) for t in range(steps + 1)]
    choices = [Bool(f"apply_G1_at_step_{t}") for t in range(steps)]
    
    # --- FIXED BOUNDARY CONDITIONS ---
    s.add(states[0] == start_idx)       # FIXED: Only the INITIAL state is bound to start_idx
    s.add(states[steps] == target_idx)  # Only the FINAL state is bound to target_idx
    
    for t in range(steps):
        current_state = states[t]
        next_state = states[t+1]
        g1_action = current_state ^ 1
        msb_is_zero = (current_state >> (num_disks - 1)) == 0
        g2_action = If(msb_is_zero, current_state ^ 3, current_state)
        s.add(next_state == If(choices[t], g1_action, g2_action))
        
    if s.check() == sat:
        m = s.model()
        # Extract the exact path as integer targets (True -> 0, False -> 1)
        return [0 if is_true(m[choices[t]]) else 1 for t in range(steps)]
    return None

# --- RUN AUTO-CORRECTING TRAIN LOOP ---
print("=== SPARK SERVER: STARTING SYSTEM DYNAMIC REINFORCEMENT WORKLOAD ===")
print("Hardware Configuration: NVIDIA GB10 Blackwell + Intel Xeon CPU\n")

steps = 2
model = SoftPolicyNetwork(steps=steps)
# Aggressive learning rate combined with soft parameters forces rapid parameter shift
optimizer = optim.SGD([model.policy_logits], lr=2.5)
criterion = nn.CrossEntropyLoss()

# Step 1: Profile current soft baseline before optimization pass
probs_before = model()
print("Baseline Network Probabilities (Before Loop):")
print(f"  Step 0: P(G1)={probs_before[0,0]:.4f}, P(G2)={probs_before[0,1]:.4f}")
print(f"  Step 1: P(G1)={probs_before[1,0]:.4f}, P(P(G2)={probs_before[1,1]:.4f}\n")

# Step 2: Query the SMT oracle for structural path correction
corrected_target_indices = run_z3_oracle(num_disks=3, steps=steps, start_idx=2, target_idx=0)

if corrected_target_indices is not None:
    print(f"Z3 Oracle Ground Truth Targets Discovered: {corrected_target_indices}")
    print("  (Index 0 = Global Swap G1, Index 1 = Branch Swap G2)\n")
    
    # Run multiple batch gradient update steps on the GPU to permanently shift the logits
    for epoch in range(5):
        optimizer.zero_grad()
        logits = model.policy_logits
        targets_tensor = torch.tensor(corrected_target_indices, device=device)
        
        loss = criterion(logits, targets_tensor)
        loss.backward()
        optimizer.step()
        
    # Step 3: Profile updated network distributions
    probs_after = model()
    print("-" * 75)
    print("Final Shifted Network Probabilities (After Neuro-Symbolic Optimization):")
    print(f"  Step 0: P(G1)={probs_after[0,0]:.4f}, P(G2)={probs_after[0,1]:.4f}")
    print(f"  Step 1: P(G1)={probs_after[1,0]:.4f}, P(G2)={probs_after[1,1]:.4f}")
    print("-" * 75)
    print("✅ SUCCESS: Continuous neural parameter space successfully updated.")
else:
    print("❌ Critical System Failure: Selected configurations cannot settle into a valid group orbit.")
