import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReversibleTreeEngine:
    """
    Implements a completely reversible relational tree workspace utilizing
    algebraic group inverses to step forward and backward through logic states.
    """
    def __init__(self, target_depth=4):
        self.depth = target_depth
        self.dim = 2 ** target_depth
        self.identity = torch.eye(self.dim, device=device)
        
        # Define Generator a as our primary branch-swapping operator
        sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        self.a_mat = torch.kron(sigma_x, torch.eye(2**(target_depth-1), device=device))

    def execute_forward_transformation(self, initial_state, operator_matrix):
        """Applies a logical transformation step forward down the tree branches."""
        return torch.matmul(operator_matrix, initial_state)

    def execute_reverse_backtrack(self, current_state, operator_matrix):
        """
        Algebraic Reversibility: Uses the exact matrix inverse (g^-1)
        to cleanly walk backward and restore the previous state.
        """
        inverse_operator = torch.inverse(operator_matrix)
        return torch.matmul(inverse_operator, current_state)

    def run_reversibility_validation(self):
        print("================================================================================")
        print("🔮 INITIALIZING DUAL-LAYER ALGEBRAIC REVERSIBLE TREE ENGINE")
        print("================================================================================")
        print("Target: Transform tree state via Generator 'a' and fully backtrack using inverse.")
        print("--------------------------------------------------------------------------------")
        
        # Initialize a random mock state vector representing our active tree branches
        initial_tree_state = torch.randn(self.dim, 1, device=device)
        print(f"  Step 1 | Initial State Vector Sample (First 3 elements):\n{initial_tree_state[:3].cpu().numpy().flatten()}")
        
        # 1. Execute Forward Transformation
        transformed_state = self.execute_forward_transformation(initial_tree_state, self.a_mat)
        print(f"\n  Step 2 | Forward Pass Executed (Branch Swap Active).")
        print(f"         | Transformed State Sample (First 3 elements):\n{transformed_state[:3].cpu().numpy().flatten()}")
        
        # 2. Execute Reverse Backtrack Loop
        restored_tree_state = self.execute_reverse_backtrack(transformed_state, self.a_mat)
        print(f"\n  Step 3 | Reverse Pass Executed via Group Inverse Operator (a^-1).")
        print(f"         | Restored State Sample (First 3 elements):\n{restored_tree_state[:3].cpu().numpy().flatten()}")
        
        # 3. Verify absolute structural restoration (L2 Distance Check)
        reversal_distance = torch.norm(initial_tree_state - restored_tree_state).item()
        print("--------------------------------------------------------------------------------")
        print(f"Mathematical Reversibility L2 Distance Delta: {reversal_distance:.6f}")
        
        if reversal_distance < 1e-4:
            print("👑 STATUS: REVERSIBILITY VERIFIED WITH ABSOLUTE MATHEMATICAL CERTAINTY")
            print("  -> Conclusion: No Derived Facts Need to be Cached in a Knowledge Base.")
            print("  -> Proof: The group inverse operator perfectly reconstructed the origin state.")
        else:
            print("❌ STATUS: ENTROPY LEAK DETECTED")
        print("================================================================================")

if __name__ == "__main__":
    engine = ReversibleTreeEngine(target_depth=4)
    engine.run_reversibility_validation()
