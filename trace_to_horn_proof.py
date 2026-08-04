import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuroSymbolicProofTranslator:
    """
    Translates raw GPU row/column matrix coordinate trajectories into 
    symbolic, first-order Horn clause resolution traces.
    """
    def __init__(self, target_depth=5):
        self.depth = target_depth
        self.dim = 2 ** target_depth
        self.identity = torch.eye(self.dim, device=device)

    def int_to_binary_path(self, index_value):
        """Converts an integer matrix index into its symbolic binary tree path."""
        # Formats the index into padding bits matching the depth of the tree
        binary_str = format(index_value, f'0{self.depth}b')
        # Map bits directly to descriptive relational directions
        return " -> ".join(["left" if bit == '0' else "right" for bit in binary_str])

    def deconstruct_to_horn_proof(self, word_name, word_matrix, max_steps=16):
        print(f"\n================================================================================")
        print(f"🔮 SYNTHESIZING PROLOG-STYLE HORN RESOLUTION FROM GPU COORDINATE TRACE")
        print(f"Target Word Composition: {word_name}")
        print(f"================================================================================")
        
        current_product = word_matrix.clone()
        
        for step in range(1, max_steps + 1):
            # 1. Capture raw hardware tensor coordinates
            active_index = torch.argmax(torch.abs(current_product)).item()
            row = active_index // self.dim
            col = active_index % self.dim
            
            # 2. Decode coordinates back to adic branch paths
            row_path = self.int_to_binary_path(row)
            col_path = self.int_to_binary_path(col)
            
            # 3. Synthesize Definite Horn Clauses representing the execution resolution
            print(f"%% Resolution Step {step:02d}:")
            print(f"proof_step(step_{step}, StateIn) :-")
            print(f"    input_node_path({row_path}),")
            print(f"    evaluated_target_path({col_path}),")
            print(f"    apply_transformation({word_name}, StateIn).")
            print(f"")
            
            if torch.allclose(current_product, self.identity, atol=1e-4):
                print(f"%% 👑 THEOREM PROVEN: Target sequence identity successfully annihilated.")
                print(f"identity_collapse({word_name}) :- proof_step(step_{step}, stable_state).")
                break
                
            current_product = torch.matmul(current_product, word_matrix)
        print("================================================================================")

if __name__ == "__main__":
    from gap_tensor_compiler import TrueBottomUpGrigorchukCompiler
    
    # Initialize the compiler and inverse proof translator
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    translator = NeuroSymbolicProofTranslator(target_depth=5)
    
    # Compute the non-commuting product: (a * b)
    word_ab = torch.matmul(compiler.a_mat, compiler.b_mat)
    
    # Execute the inverse logical translation
    translator.deconstruct_to_horn_proof("[a, b]", word_ab, max_steps=16)
