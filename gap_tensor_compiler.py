import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrueBottomUpGrigorchukCompiler:
    """
    Inductively constructs the Grigorchuk group generators from depth 1 up to N
    to ensure perfect, uncorrupted fractal sequence properties on GPUs.
    """
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        # Build the final full-dimensional representations
        self.a_mat, self.b_mat, self.c_mat, self.d_mat = self._build_tree_layer(max_depth)
        self.dim = 2 ** max_depth

    def _build_tree_layer(self, current_depth):
        # Base Case: At depth 1, the space has 2 states (0 and 1)
        if current_depth == 1:
            a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
            b = torch.eye(2, device=device)
            c = torch.eye(2, device=device)
            d = torch.eye(2, device=device)
            return a, b, c, d

        # Inductive Step: Build child definitions first
        sub_depth = current_depth - 1
        sub_dim = 2 ** sub_depth
        dim_local = 2 ** current_depth
        
        a_sub, b_sub, c_sub, d_sub = self._build_tree_layer(sub_depth)

        # 1. 'a' always acts as a pure permutation swap at the current tree split
        sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        a_local = torch.kron(sigma_x, torch.eye(sub_dim, device=device))

        # 2. b = (a, c) -> Assemble into block-diagonal wreath product
        b_local = torch.zeros(dim_local, dim_local, device=device)
        b_local[:sub_dim, :sub_dim] = a_sub
        b_local[sub_dim:, sub_dim:] = c_sub

        # 3. c = (a, d) -> Assemble into block-diagonal wreath product
        c_local = torch.zeros(dim_local, dim_local, device=device)
        c_local[:sub_dim, :sub_dim] = a_sub
        c_local[sub_dim:, sub_dim:] = d_sub

        # 4. d = (1, b) -> Assemble into block-diagonal wreath product
        d_local = torch.zeros(dim_local, dim_local, device=device)
        d_local[:sub_dim, :sub_dim] = torch.eye(sub_dim, device=device)
        d_local[sub_dim:, sub_dim:] = b_sub

        return a_local, b_local, c_local, d_local

    def compute_word_order(self, word_matrix, max_order=32):
        """Simulates GAP's Order() function by evaluating identity collapse."""
        current_product = word_matrix.clone()
        identity = torch.eye(self.dim, device=device)
        
        for order in range(1, max_order + 1):
            if torch.allclose(current_product, identity, atol=1e-4):
                return order
            current_product = torch.matmul(current_product, word_matrix)
        return -1

if __name__ == "__main__":
    print("================================================================================")
    print("🔮 INDUCTIVE GAP TENSOR COMPILER & ORDER VERIFICATION RUNNER")
    print("================================================================================")
    
    # Initialize compiler at depth 5
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    print("🚀 SUCCESS: Inductively constructed fractal Grigorchuk generator tensors.")
    
    # Execute the non-commuting composition: (a * b)
    word_ab = torch.matmul(compiler.a_mat, compiler.b_mat)
    computed_order = compiler.compute_word_order(word_ab, max_order=32)
    
    print(f"\nExecuting Order Check: Order(a * b);")
    print(f"  -> True Analytical Value: 16")
    print(f"  -> GPU Tensor Computed Order: {computed_order}")
    print("================================================================================")
