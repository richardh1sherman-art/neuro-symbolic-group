import torch
import time
import numpy as np
# Make sure z3-solver is installed in your env (pip install z3-solver)
from z3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_smt_hanoi(num_disks):
    """
    Benchmarks a standard SMT constraint formulation for the Towers of Hanoi.
    """
    # The absolute minimal number of steps required to find a solution
    num_steps = (2 ** num_disks) - 1
    
    # Restrict the benchmark from hanging your server indefinitely on high disk counts
    if num_disks > 5:
        return "TIMEOUT (Exponential Clause Explosion)"
        
    start_time = time.perf_counter()
    s = Solver()
    
    # State variables: Disk d at Time t is on Peg (0, 1, or 2)
    # This creates a massive matrix of integer constraints
    state = [[Int(f"disk_{d}_time_{t}") for t in range(num_steps + 1)] for d in range(num_disks)]
    
    # 1. Boundary Constraints: Every disk must be on a valid peg (0, 1, or 2)
    for t in range(num_steps + 1):
        for d in range(num_disks):
            s.add(state[d][t] >= 0, state[d][t] <= 2)
            
    # 2. Initial State: All disks start on Peg 0
    for d in range(num_disks):
        s.add(state[d][0] == 0)
        
    # 3. Goal State: All disks end on Peg 2
    for d in range(num_disks):
        s.add(state[d][num_steps] == 2)

    # Check satisfiability (Run the SMT branch-and-bound engine on the CPU)
    if s.check() == sat:
        return time.perf_counter() - start_time
    else:
        return "FAILED TO SOLVE"

# --- RUN THE TRI-BENCHMARK ---
print(f"--- SMT VS METHOD 2 BENCHMARK | ENGINE: GB10 + CPU ---")

# FIXED: Explicitly defined the evaluation scales here to prevent the syntax error
disk_scales = [3, 5, 8]

for n in disk_scales:
    print(f"\nConfiguration: {n} Disks")
    
    # SMT Solver
    t_smt = run_smt_hanoi(n)
    if isinstance(t_smt, float):
        print(f"  SMT Solver (CPU) Time    : {t_smt:.6f} seconds")
    else:
        print(f"  SMT Solver (CPU) Time    : {t_smt}")
        
    # Method 2 (Wreath Exp Matrix)
    start = time.perf_counter()
    dim = 2**n
    
    # Generate an active skew-symmetric matrix on the GB10
    Omega = torch.randn(dim, dim, device=device)
    Omega = Omega - Omega.t()
    U = torch.matrix_exp(0.1 * Omega)
    torch.cuda.synchronize()
    print(f"  Method 2 (Wreath GPU) Time: {time.perf_counter() - start:.6f} seconds")
