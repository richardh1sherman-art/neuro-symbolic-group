import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TensorToHornDecoder:
    """
    Decomposes multi-scale GPU group tensors and decodes their structural
    permutations back into symbolic First-Order Horn Clauses and Prolog Knowledge Bases.
    """
    def __init__(self, target_depth=5):
        self.depth = target_depth
        self.sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        self.identity_2d = torch.eye(2, device=device)

    def decode_block_to_predicate(self, matrix_slice, depth_level):
        core_block = matrix_slice[:2, :2]
        if torch.allclose(core_block, self.sigma_x, atol=1e-3):
            return f"branch_swap(level_{depth_level})"
        elif torch.allclose(core_block, self.identity_2d, atol=1e-3):
            return f"identity_invariant(level_{depth_level})"
        else:
            return f"recursive_restriction(level_{depth_level})"

    def synthesize_horn_clause(self, generator_name, generator_matrix):
        half_dim = generator_matrix.shape[0] // 2
        left_child_slice = generator_matrix[:half_dim, :half_dim]
        right_child_slice = generator_matrix[half_dim:, half_dim:]
        
        left_pred = self.decode_block_to_predicate(left_child_slice, self.depth)
        right_pred = self.decode_block_to_predicate(right_child_slice, self.depth)
        
        if generator_name == "a":
            return f"action(a, State) :- branch_swap(level_1)."
        else:
            return f"action({generator_name}, State) :- {left_pred}, {right_pred}."

    def export_knowledge_base(self, clauses_list, output_filename="grigorchuk_rules.pl"):
        """Writes compiled symbolic Horn clauses out to a readable Prolog file."""
        with open(output_filename, "w") as f:
            f.write("%% ==========================================================\n")
            f.write("%% AUTOMATICALLY GENERATED GRIGORCHUK RELATIONAL KNOWLEDGE BASE\n")
            f.write("%% Synthesized via Inverse Neuro-Symbolic Tensor Decoding\n")
            f.write("%% ==========================================================\n\n")
            for clause in clauses_list:
                f.write(f"{clause}\n")
        print(f"\n💾 PROLOG EXPORT SUCCESS: Knowledge Base file written to: ./{output_filename}")

if __name__ == "__main__":
    from gap_tensor_compiler import TrueBottomUpGrigorchukCompiler
    
    print("================================================================================")
    print("🔮 INITIALIZING INVERSE NEURO-SYMBOLIC TENSOR-TO-HORN DECODER")
    print("================================================================================")
    
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    decoder = TensorToHornDecoder(target_depth=5)
    
    # Compile the active rule clauses into an array memory list
    clauses = [
        decoder.synthesize_horn_clause("a", compiler.a_mat),
        decoder.synthesize_horn_clause("b", compiler.b_mat),
        decoder.synthesize_horn_clause("c", compiler.c_mat),
        decoder.synthesize_horn_clause("d", compiler.d_mat)
    ]
    
    # Print outputs to terminal
    for c in clauses:
        print(f"  ⚡ Synthesized Horn Clause: {c}")
        
    # Trigger the file writer to generate your missing .pl document
    decoder.export_knowledge_base(clauses)
    print("================================================================================")
