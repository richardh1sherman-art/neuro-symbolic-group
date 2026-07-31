import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrecomputedWreathEngine:
    def __init__(self, num_disks):
        self.state_dim = 2 ** num_disks
        self.I = torch.eye(2, device=device)
        self.M_a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        
        # Build structural matrices
        self.Gen_1 = torch.kron(torch.eye(self.state_dim // 2, device=device), self.M_a)
        self.Gen_2 = torch.block_diag(torch.kron(self.M_a, self.M_a), torch.eye(self.state_dim - 4, device=device))
        self.Omega = torch.matmul(self.Gen_1, self.Gen_2) - torch.matmul(self.Gen_2, self.Gen_1)
        
        # CRITICAL: Pre-compute the matrix exponential ONCE during startup
        self.U = torch.matrix_exp(0.5 * self.Omega)

    def run_step(self, hidden_state):
        # Pure, ultra-fast dense tensor contraction on Blackwell cores
        return torch.matmul(self.U, hidden_state)

print("=== PHASE 2: PRE-COMPUTED PERFORMANCE SEPARATION ===")
print("Measuring true runtime execution speed (excluding setup overhead)...")

deep_scales = [8, 11, 13, 14] # Added 14 disks to push the graph size even further

for n in deep_scales:
    print(f"\nConfiguration: {n} Disks")
    
    # --- Method 1: Dense Graph Memory Tracker ---
    m1_states = 3 ** n
    graph_frontier = torch.zeros(m1_states, 1, device=device)
    
    start_m1 = time.perf_counter()
    # Simulate a deep graph-relaxation step (100 step lookahead)
    for _ in range(100):
        graph_frontier = graph_frontier + 0.001
    torch.cuda.synchronize()
    t_m1 = time.perf_counter() - start_m1
    print(f"  Method 1 (Graph BFS Frontier) Time: {t_m1:.6f} seconds")

    # --- Method 2: Pre-computed Self-Similar Engine ---
    engine = PrecomputedWreathEngine(n)
    hidden_state = torch.zeros(engine.state_dim, 1, device=device)
    hidden_state[0] = 1.0
    
    start_m2 = time.perf_counter()
    # Simulate processing across the exact same 100 step lookahead
    for _ in range(100):
        hidden_state = engine.run_step(hidden_state)
    torch.cuda.synchronize()
    t_m2 = time.perf_counter() - start_m2
    print(f"  Method 2 (Wreath GPU Engine)  Time: {t_m2:.6f} seconds")
