import torch
import time
import json
import os

# Target the active hardware accelerator core
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfSimilarWreathAutomaton:
    """
    Simulates a compiler core that scales group generators 
    fractally based on depth limits.
    """
    def __init__(self, depth):
        self.depth = depth
        self.dim = 2 ** depth
        
        # Primitive root permutation tensor (Isomorphic to GAP's binary alphabet swap)
        self.sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        
        # Compile a static generator utilizing structural Kronecker math
        self.G = torch.kron(self.sigma_x, torch.eye(2**(depth-1), device=device)) if depth > 0 else torch.eye(1, device=device)

    def execute_contraction(self, state_vector):
        # Bi-reversible involution pass: G * state
        return torch.matmul(self.G, state_vector)

def run_automated_suite():
    print("================================================================================")
    print("🚀 INITIALIZING SELF-SIMILAR AUTOMATA DEPTH CEILING BENCHMARK SUITE")
    print(f"Accelerator Core: {device} | Status: Live Verification Loop")
    print("================================================================================")
    
    # Test bounds across progressive tree depths (d from 2 to 14 layers)
    depth_horizons = [2, 4, 6, 8, 10, 12, 14]
    benchmark_logs = []

    for d in depth_horizons:
        # Prevent VRAM overflow on ultra-deep trees by allocating structured states
        try:
            # Step 1: Initialize the group math core at depth d
            t_init_0 = time.perf_counter()
            automaton = SelfSimilarWreathAutomaton(depth=d)
            init_time = time.perf_counter() - t_init_0
            
            # Step 2: Generate an active dense feature vector (superposition state)
            input_psi = torch.randn(automaton.dim, 1, device=device)
            
            # Step 3: Profile execution speed over 50 iterations to ensure statistical normalization
            iterations = 50
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            t_run_0 = time.perf_counter()
            for _ in range(iterations):
                output_psi = automaton.execute_contraction(input_psi)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            avg_exec_time = (time.perf_counter() - t_run_0) / iterations
            
            # Track memory footprint parameters
            allocated_vram = torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
            
            print(f"Depth ceiling (d): {d:2d} | Dimension: {automaton.dim:6d} | Init: {init_time:.5f}s | Run: {avg_exec_time:.6f}s | VRAM: {allocated_vram:.2f}MB")
            
            benchmark_logs.append({
                "depth_ceiling": d,
                "dimension": automaton.dim,
                "init_seconds": init_time,
                "exec_seconds": avg_exec_time,
                "vram_mb": allocated_vram
            })
            
        except RuntimeError as e:
            print(f"Depth ceiling (d): {d:2d} | FAILED (System Memory Ceiling Exceeded)")
            benchmark_logs.append({"depth_ceiling": d, "status": "OOM_LIMIT"})
            break

    # Export metrics package straight to local storage
    os.makedirs("./results", exist_ok=True)
    with open("./results/depth_benchmarks.json", "w") as f:
        json.dump(benchmark_logs, f, indent=2)
    print("--------------------------------------------------------------------------------")
    print("✅ Benchmark package logged cleanly to JSON: ./results/depth_benchmarks.json")
    print("================================================================================")

if __name__ == "__main__":
    run_automated_suite()
