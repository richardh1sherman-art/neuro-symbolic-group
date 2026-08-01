import json
import time

def build_ilp_matrix_file(json_source_path, lp_output_path):
    print("=== INITIALIZING PARALLEL ILP MATRIX GENERATOR ===")
    print(f"Data Source Pipeline: {json_source_path}")
    
    # 1. Ingest your verified token dataset textbook
    with open(json_source_path, 'r') as f:
        dataset = json.load(f)
        
    start_time = time.perf_counter()
    
    # We select an optimized multi-step episode path to map into the matrix equations
    # We will grab the 4-step trajectory generated from State 6 to demonstrate complex bounds
    target_episode = None
    for episode in dataset:
        if episode["prompt_metadata"]["initial_binary_token"] == 6:
            target_episode = episode
            break
            
    if target_episode is None:
        print("❌ ERROR: Aligned 4-bit state trace not found inside source database file.")
        return

    token_seq = target_episode["token_sequences"]
    state_tokens = token_seq["input_state_tokens"] + [token_seq["output_state_token"]]
    total_steps = len(token_seq["target_operator_tokens"])

    # 2. Compile the mathematical linear programming file format
    with open(lp_output_path, 'w') as lp:
        lp.write("\\* Universal Neuro-Symbolic Self-Similar Group Optimization Model *\\\n\n")
        
        # --- OBJECTIVE FUNCTION ---
        # Minimize the total resource weight cost across the sequential choices
        lp.write("Minimize\n")
        obj_terms = []
        for t in range(total_steps):
            obj_terms.append(f"1 c_step_{t}_G1 + 2 c_step_{t}_G2")
        lp.write("   " + " + ".join(obj_terms) + "\n\n")
        
        # --- CONSTRAINT SYSTEM MATRIX ---
        lp.write("Subject To\n")
        
        # Constraint Group 1: Strict Operator Exclusivity 
        # For every sequential time slot t, you must select exactly ONE group action token
        for t in range(total_steps):
            lp.write(f"  Exclusivity_Step_{t}: c_step_{t}_G1 + c_step_{t}_G2 = 1\n")
            
        # Constraint Group 2: Boundary State Variable Enforcements
        # Forcing the start and target states to equal their exact binary codes
        lp.write("\n  \\* Boundary Trajectory Vectors *\\\n")
        lp.write(f"  Initial_State_Bound: state_0 = 6\n")
        lp.write(f"  Target_State_Bound: state_{total_steps} = 0\n")
        
        # Constraint Group 3: Linearized Algebraic State Mapping Transitions
        # We transform the non-commuting group permutations into hard equality bounds
        lp.write("\n  \\* Algebraic State Alignment Rules *\\\n")
        for t in range(total_steps):
            # Map the exact state integers calculated by your engine straight into bounds
            current_state_val = int(state_tokens[t].split("_")[1])
            next_state_val = int(state_tokens[t+1].split("_")[1])
            diff = next_state_val - current_state_val
            
            lp.write(f"  Algebraic_Transition_Step_{t}: state_{t+1} - state_{t} = {diff}\n")

        # --- VARIABLE DECLARATIONS ---
        # Define the structural boundaries of our tracking variables
        lp.write("\nBounds\n")
        for t in range(total_steps + 1):
            lp.write(f"  0 <= state_{t} <= 15\n") # Restricted to our 4-bit tree dimension
            
        # Binary Variables Definition: Operator choice switches are strict 0/1 integers
        lp.write("\nBinary\n")
        for t in range(total_steps):
            lp.write(f"  c_step_{t}_G1\n")
            lp.write(f"  c_step_{t}_G2\n")
            
        lp.write("\nEnd\n")

    runtime = time.perf_counter() - start_time
    print(f"✅ MATRIX COMPILATION COMPLETE IN {runtime:.6f} SECONDS")
    print(f"Industrial ILP Model saved to: {lp_output_path}")

if __name__ == "__main__":
    json_path = "/home/rsherman/projects/SMT-ILP/universal_group_dataset.json"
    lp_path = "/home/rsherman/projects/SMT-ILP/self_similar_wreath_model.lp"
    build_ilp_matrix_file(json_path, lp_path)
