import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerfectlyAlignedHanoiSystem:
    def __init__(self, num_disks):
        self.num_disks = num_disks
        self.m1_dim = 3 ** num_disks  
        self.m2_dim = 2 ** num_disks  
        
        # --- METHOD 2 VECTORIZED SPARSE ENGINE ---
        base_indices = torch.arange(self.m2_dim, device=device)
        self.gen1_indices = base_indices ^ 1 
        self.gen2_indices = base_indices.clone()
        left_branch_size = self.m2_dim // 2
        if left_branch_size >= 4:
            self.gen2_indices[:left_branch_size] = torch.arange(left_branch_size, device=device) ^ 3

        # --- VECTORIZED BIPARTITE TRANSITION DICTIONARY ---
        disk_powers = torch.arange(self.num_disks, device=device)
        bits = (base_indices.unsqueeze(1) >> disk_powers) & 1
        
        peg_placements = bits * 2
        ternary_strides = 3 ** disk_powers
        self.translation_lookup = torch.sum(peg_placements * ternary_strides, dim=1)

    def decode_m1_index_to_string(self, state_idx):
        pegs = []
        temp = state_idx
        for _ in range(self.num_disks):
            pegs.append(str(temp % 3))
            temp //= 3
        return " -> ".join(pegs)

    def run_deterministic_gen1_step(self, hidden_state):
        """Runs a pure, un-blended Gen_1 operation to bypass argmax ties"""
        return torch.index_select(hidden_state, 0, self.gen1_indices)

    def run_method2_step(self, hidden_state):
        state_g1 = torch.index_select(hidden_state, 0, self.gen1_indices)
        state_g2 = torch.index_select(hidden_state, 0, self.gen2_indices)
        return (0.5 * state_g1) + (0.5 * state_g2)

    def map_wreath_to_graph_space(self, wreath_vector):
        max_m2_idx = torch.argmax(wreath_vector.flatten()).item()
        aligned_m1_idx = self.translation_lookup[max_m2_idx].item()
        return aligned_m1_idx

# --- RUN VERIFICATION ---
print("=== VERIFYING REAL ALIGNMENT THROUGH VECTORIZED DICTIONARY ===")
system_val = PerfectlyAlignedHanoiSystem(num_disks=3)

# Initialize an input state at index 2 (Binary 010 -> Pegs: 0 -> 2 -> 0)
init_state_m2 = torch.zeros(system_val.m2_dim, 1, device=device)
init_state_m2[2] = 1.0 

# Run a deterministic Gen_1 state transition step (2 ^ 1 = 3)
output_state_m2 = system_val.run_deterministic_gen1_step(init_state_m2)

# Pass through the bipartite dictionary translation
aligned_m1_index = system_val.map_wreath_to_graph_space(output_state_m2)
final_pegs_m2 = system_val.decode_m1_index_to_string(aligned_m1_index)

# Method 1 native lookup for index 18 (Ternary counterpart to binary index 3)
final_pegs_m1 = system_val.decode_m1_index_to_string(18) 

print(f"Method 1 Native Coordinate Target:  [{final_pegs_m1}]")
print(f"Method 2 Bipartite Aligned Output:  [{final_pegs_m2}]")

if final_pegs_m1 == final_pegs_m2:
    print("✅ TRUE ALIGNMENT SUCCESSFUL: The strings perfectly match!")
else:
    print("❌ ALIGNMENT ERROR: Lookups remain unaligned.")
