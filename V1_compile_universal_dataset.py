import torch
import json
import time
from z3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UniversalBipartiteMapper:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.state_dim = 2 ** num_bits
        
        base_indices = torch.arange(self.state_dim, device=device)
        disk_powers = torch.arange(self.num_bits, device=device)
        
        # Fixed endianness alignment to match physical coordinates
        flipped_powers = torch.flip(disk_powers, dims=[0])
        bits = (base_indices.unsqueeze(1) >> flipped_powers) & 1
        
        peg_placements = bits * 2
        ternary_strides = 3 ** disk_powers
        self.translation_lookup = torch.sum(peg_placements * ternary_strides, dim=1)

    def get_schreier_string(self, binary_idx):
        m1_idx = self.translation_lookup[binary_idx].item()
        pegs = []
        temp = m1_idx
        for _ in range(self.num_bits):
            pegs.append(str(temp % 3))
            temp //= 3
        return " -> ".join(pegs)

def query_smt_oracle(num_bits, steps, start_idx, target_idx):
    s = Solver()
    states = [BitVec(f"state_{t}", num_bits) for t in range(steps + 1)]
    choices = [Bool(f"apply_G1_at_step_{t}") for t in range(steps)]
    
    s.add(states[0] == start_idx)       
    s.add(states[steps] == target_idx)
    
    for t in range(steps):
        curr_state = states[t]
        next_state = states[t+1]
        
                # Generator 1: Global LSB swap (toggles the smallest disk)
        g1_action = curr_state ^ 1
        
        # --- THE DEFINTIVE FRACTAL LOGIC FIX ---
        # Generator 2 is a strict recursive Hanoi operator.
        # If the lower bit is active, it mutates the middle bit (^2).
        # If the lower bit is passive, it mutates the higher bit (^4).
        # This completely eliminates the A ^ A = 0 teleportation bug!
        condition_active = (curr_state & 1) == 1
        g2_action = If(condition_active, curr_state ^ 2, curr_state ^ 4)
        
        s.add(next_state == If(choices[t], g1_action, g2_action))


        
    if s.check() == sat:
        m = s.model()
        op_sequence = [1 if is_true(m[choices[t]]) else 2 for t in range(steps)]
        state_sequence = [m[states[t]].as_long() for t in range(steps + 1)]
        return op_sequence, state_sequence
    return None, None

# --- SCALE UP THE WORKLOAD ---
print("=== SPARK SERVER: INITIALIZING DEEP WORKLOAD COMPILER ===")
print("Hardware Architecture: NVIDIA GB10 Blackwell | Space: 4-Bit Wreath")

num_bits = 4   # Scaled to 4 bits (16 unique structural states)
max_steps = 8  # Expanded search horizon to clear deep parity blocks
target_state = 0

mapper = UniversalBipartiteMapper(num_bits=num_bits)
dataset_matrix = []

start_compile_time = time.perf_counter()

# Sweep through all 16 states
for start_state in range(2 ** num_bits):
    if start_state == target_state:
        continue
        
    for steps in range(1, max_steps + 1):
        op_seq, state_seq = query_smt_oracle(num_bits, steps, start_state, target_state)
        
        if op_seq is not None:
            schreier_trace = [mapper.get_schreier_string(s_idx) for s_idx in state_seq]
            
            episode_entry = {
                "prompt_metadata": {
                    "num_bits": num_bits,
                    "initial_binary_token": start_state,
                    "target_binary_token": target_state,
                    "total_operational_steps": steps
                },
                "token_sequences": {
                    "input_state_tokens": [f"STATE_{s}" for s in state_seq[:-1]],
                    "target_operator_tokens": [f"OP_G{o}" for o in op_seq],
                    "output_state_token": f"STATE_{state_seq[-1]}"
                },
                "schreier_graph_projection": schreier_trace
            }
            dataset_matrix.append(episode_entry)
            print(f"  Processed Prompt State {start_state}: Found valid {steps}-step orbit path.")
            break 

# Save dataset
output_path = "/home/rsherman/projects/SMT-ILP/universal_group_dataset.json"
with open(output_path, "w") as f:
    json.dump(dataset_matrix, f, indent=4)

print("\n" + "="*70)
print(f"✅ COMPILATION COMPLETE IN {time.perf_counter() - start_compile_time:.4f} SECONDS")
print(f"Database Matrix Saved to: {output_path}")
print(f"Total Unique Compiled Training Episodes: {len(dataset_matrix)}")
print("="*70)
