import torch
import torch.nn.functional as F
import time
from pixel2pixel_self_similar import Pixel2PixelNeuroSymbolicDSL
from execute_recursive_predicate import RecursiveILPPredicateEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def execute_cross_domain_bridge():
    print("=== SPARK SERVER: INITIALIZING CROSS-DOMAIN HIERARCHICAL BRIDGE ===")
    print(f"Accelerating Integrated Visual-Relational Workload on Device: {device}\n")
    
    # 1. Initialize the Image Processing Core (Cantor Quadtree Space)
    p2p_system = Pixel2PixelNeuroSymbolicDSL(levels_N=3)
    mock_noisy_image = torch.randn(1, 1, 64, 64, device=device)
    
    print("Step 1: Running Zero-Shot Pixel2Pixel Quadtree Contraction...")
    dsl_string, denoised_pixel_bank = p2p_system(mock_noisy_image)
    
    # Compress the spatial image bank down to a 32-dimensional matrix slice
    spatial_features = F.adaptive_avg_pool2d(denoised_pixel_bank, (32, 1))
    
    # --- FIXED TRANSPOSE AND SHAPE LAYOUT ALIGNMENT ---
    # Re-aligns [1, 1, 32] cleanly into a standard 2D vector [32, 1]
    spatial_features_psi = spatial_features.view(-1, 1)
    print(f"  Visual Vector Compiled. Aligned Vector Layout Shape: {list(spatial_features_psi.shape)}")
    
    # 2. Initialize the Inductive Logic Programming Predicate Engine (5-Bit Tree)
    ilp_engine = RecursiveILPPredicateEngine(max_depth=5)
    
    print("\nStep 2: Injecting Visual Feature Tensors directly into Recursive Predicate Matrix...")
    t0 = time.perf_counter()
    output_logic_psi = ilp_engine.evaluate_predicate_recursion(spatial_features_psi)
    torch.cuda.synchronize()
    total_speed = time.perf_counter() - t0
    
    print("-" * 80)
    print("Fused Visual-Logical Pipeline Cleared Execution Loops Successfully:")
    print(f"  Total Cross-Domain End-to-End Processing Speed : {total_speed:.6f} seconds")
    print(f"  Final Hardened Predicate State Tensor Shape    : {list(output_logic_psi.shape)}")
    print("================================================================================")

if __name__ == "__main__":
    execute_cross_domain_bridge()
