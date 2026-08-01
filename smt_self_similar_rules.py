import time
from z3 import *

def solve_hanoi_symbolic_dynamic(num_disks, start_state=2, target_state=0, max_steps=10):
    print(f"=== Z3 DYNAMIC SYMBOLIC WREATH SOLVER ({num_disks} Disks) ===")
    print(f"Target: Transition from Binary State {start_state} -> {target_state}\n")
    
    # Iteratively expand the temporal step window to find valid group orbits
    for steps in range(1, max_steps + 1):
        s = Solver()
        
        # Instantiate symbolic bit-vectors for this step depth
        states = [BitVec(f"state_{t}", num_disks) for t in range(steps + 1)]
        choices = [Bool(f"apply_G1_at_step_{t}") for t in range(steps)]
        
        # Enforce boundary configurations
        s.add(states[0] == start_state)
        s.add(states[steps] == target_state)
        
        # Inject the self-similar group algebraic rules
        for t in range(steps):
            current_state = states[t]
            next_state = states[t+1]
            
            # G1 Action (Global Swap)
            g1_action = current_state ^ 1
            
            # G2 Action (Conditional Block Diagonal Left-Branch Swap)
            msb_is_zero = (current_state >> (num_disks - 1)) == 0
            g2_action = If(msb_is_zero, current_state ^ 3, current_state)
            
            # Tie the next state strictly to the path choice variable
            s.add(next_state == If(choices[t], g1_action, g2_action))
            
        # Check if the group equations balance at this step depth
        start_time = time.perf_counter()
        if s.check() == sat:
            runtime = time.perf_counter() - start_time
            print(f"✅ SOLUTION FOUND at Step Depth: {steps} (Solved in {runtime:.6f} seconds)")
            
            m = s.model()
            print("\nDiscovered Reversible Operator Plan:")
            for t in range(steps):
                state_val = m[states[t]].as_long()
                op_used = "G1 (Global Swap)" if is_true(m[choices[t]]) else "G2 (Branch Swap)"
                print(f"  Step {t}: State Index {state_val} (Binary: {state_val:03b}) -> Apply {op_used}")
            
            final_val = m[states[steps]].as_long()
            print(f"  Final Destination: State Index {final_val} (Binary: {final_val:03b})")
            return True
            
        print(f"  Step Depth {steps}: UNSAT (Orbit Parity Blocked)")
        
    print("\n❌ Path limit exceeded. No valid group combinations found.")
    return False

# Run the dynamic solver
solve_hanoi_symbolic_dynamic(num_disks=3, start_state=2, target_state=0)
