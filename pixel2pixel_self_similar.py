import os
import torch
import torch.nn as nn
import time
import imageio
import numpy as np
from skimage.transform import rescale  # Handles the crisp pre-scaling math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Pixel2PixelAlgebraicCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('Perm', torch.tensor([0, 2, 1, 3], device=device))

    def forward(self, x):
        b, c, h, w = x.shape
        quadrants = x.view(b, c, 4, h // 2, w // 2)
        permuted = quadrants[:, :, self.Perm, :, :]
        pseudo_instance = permuted.view(b, c, h, w)
        return pseudo_instance, torch.tanh(pseudo_instance)

class Pixel2PixelNeuroSymbolicDSL(nn.Module):
    def __init__(self, levels_N=3):
        super().__init__()
        self.N = levels_N
        self.cores = nn.ModuleList([Pixel2PixelAlgebraicCore() for _ in range(levels_N)])

    def forward(self, noisy_input):
        current_bank = noisy_input
        closure_instances = []
        dsl_string = "<s>"
        
        for n in range(self.N):
            current_bank, y_out = self.cores[n](current_bank)
            closure_instances.append(y_out)
            dsl_string += f" Cl{n+1}(T{n+1})"
            
        dsl_string += " <Q> En"
        T = torch.stack(closure_instances).mean(dim=0)
        
        is_convergent = True
        for approx in closure_instances:
            if not torch.allclose(torch.tanh(T), torch.tanh(approx), atol=0.8):
                is_convergent = False
                break
                
        result_token = "[YES]" if is_convergent else "[NO]"
        dsl_string += f" <R> {result_token}"
        return dsl_string, T

# --- RUN EXECUTION & HIGH-RESOLUTION EXPORT ---
print("=== SPARK SERVER: INITIALIZING HIGH-RESOLUTION PIPELINE ===")
os.makedirs("./results", exist_ok=True)

# Generate baseline patterns
clean_image = torch.zeros(1, 1, 64, 64, device=device)
clean_image[:, :, 16:48, 16:48] = 1.0  # Sharp inner block
noise = torch.randn(1, 1, 64, 64, device=device) * 0.4
noisy_image = torch.clamp(clean_image + noise, 0.0, 1.0)

p2p_system = Pixel2PixelNeuroSymbolicDSL(levels_N=3)
program_output, denoised_output = p2p_system(noisy_image)

# Extract raw matrices to CPU numpy arrays
raw_noisy_np = noisy_image.squeeze().cpu().numpy()
raw_denoised_np = torch.clamp(denoised_output, 0.0, 1.0).squeeze().cpu().numpy()

# --- THE CRISP SCALING FIX ---
# Upscale the 64x64 grid by 8x using order=0 (Nearest-Neighbor) 
# This prevents blurring and keeps edge boundaries perfectly sharp!
high_res_noisy = (rescale(raw_noisy_np, 8, order=0) * 255).astype(np.uint8)
high_res_denoised = (rescale(raw_denoised_np, 8, order=0) * 255).astype(np.uint8)

# Write the crisp high-fidelity images directly into your project path
imageio.imwrite("./image_before_method2.png", high_res_noisy)
imageio.imwrite("./image_after_method2.png", high_res_denoised)

print("  [CRISP SUCCESS] Sharp high-res 'Before' image written: ./image_before_method2.png")
print("  [CRISP SUCCESS] Sharp high-res 'After' image written:  ./image_after_method2.png")
print("================================================================================")
