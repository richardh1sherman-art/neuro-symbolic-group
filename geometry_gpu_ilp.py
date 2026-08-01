```````import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class DifferentiableGeometryILP(nn.Module):
    """
    Continuous Thought Inductive Logic Engine tailored for Interval and Half-Plane domains.
    Optimizes real-valued coordinate vectors and predicate weights simultaneously on CUDA.
    """
    def __init__(self, mode="halfplane"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        
        if self.mode == "interval":
            # Learn optimal real-valued lower and upper constraint values directly in continuous space
            self.min_val = nn.Parameter(torch.tensor([0.0], device=self.device))
            self.max_val = nn.Parameter(torch.tensor([1.0], device=self.device))
        else:
            # Learn optimal halfplane boundary weights: a*x + y <= b
            self.a = nn.Parameter(torch.tensor([1.0], device=self.device))
            self.b = nn.Parameter(torch.tensor([0.0], device=self.device))

    def forward(self, pos_coords, neg_coords):
        loss = 0.0
        
        if self.mode == "interval":
            # Constraint: min_val <= x <= max_val
            for x in pos_coords:
                tx = torch.tensor([x], dtype=torch.float32, device=self.device)
                # Minimize out-of-bounds violations using smooth continuous ReLU layers
                loss += torch.clamp(self.min_val - tx, min=0.0) ** 2
                loss += torch.clamp(tx - self.max_val, min=0.0) ** 2
            for x in neg_coords:
                tx = torch.tensor([x], dtype=torch.float32, device=self.device)
                # Force negative examples outside the coordinate bounds
                loss += torch.clamp(4.0 - (torch.clamp(self.min_val - tx, min=0.0)**2 + torch.clamp(tx - self.max_val, min=0.0)**2), min=0.0)
                
        elif self.mode == "halfplane":
            # Constraint: a*x + y <= b  ->  a*x + y - b <= 0
            for x, y in pos_coords:
                tx = torch.tensor([x], dtype=torch.float32, device=self.device)
                ty = torch.tensor([y], dtype=torch.float32, device=self.device)
                violation = self.a * tx + ty - self.b
                loss += torch.clamp(violation, min=0.0) ** 2
            for x, y in neg_coords:
                tx = torch.tensor([x], dtype=torch.float32, device=self.device)
                ty = torch.tensor([y], dtype=torch.float32, device=self.device)
                violation = self.a * tx + ty - self.b
                loss += torch.clamp(2.0 - torch.clamp(violation, min=0.0), min=0.0) ** 2
                
        return loss

def execute_geometry_induction():
    print("🚀 Booting Real-Valued Geometry Superposition Engine on GPU Core...")
    
    # Simulate generating the exact Problem data structures from your script configurations
    # (30 positive coordinate points, 30 negative coordinate points)
    np.random.seed(2)
    
    # 1. Evaluate the Interval Problem
    print("\n[DOMAIN: INTERVAL] Inducing numerical boundary constraints...")
    pos_interval = np.random.uniform(2.0, 8.0, 30).tolist()
    neg_interval = np.random.uniform(-1.0, 1.9, 15).tolist() + np.random.uniform(8.1, 12.0, 15).tolist()
    
    model_int = DifferentiableGeometryILP(mode="interval")
    opt_int = optim.Adam(model_int.parameters(), lr=0.05)
    
    t0 = time.perf_counter()
    for _ in range(1500):
        opt_int.zero_grad()
        loss = model_int(pos_interval, neg_interval)
        if loss.item() < 1e-4: break
        loss.backward()
        opt_int.step()
        
    print(f"  Interval Induction Complete in: {time.perf_counter() - t0:.4f}s")
    print(f"  Induced Symbolic Rule: {model_int.min_val.item():.3f} <= X <= {model_int.max_val.item():.3f}")

    # 2. Evaluate the Halfplane Problem (Matrix Inequality)
    print("\n[DOMAIN: HALFPLANE] Inducing spatial slope orientations...")
    # Setup coordinates: True system uses target coefficients (e.g., a=2.5, b=4.0)
    pos_halfplane = [ [x, 4.0 - 2.5*x - np.random.uniform(0, 2)] for x in np.random.uniform(-2, 2, 30) ]
    neg_halfplane = [ [x, 4.0 - 2.5*x + np.random.uniform(1, 3)] for x in np.random.uniform(-2, 2, 30) ]
    
    model_hp = DifferentiableGeometryILP(mode="halfplane")
    opt_hp = optim.Adam(model_hp.parameters(), lr=0.05)
    
    t0 = time.perf_counter()
    for _ in range(1500):
        opt_hp.zero_grad()
        loss = model_hp(pos_halfplane, neg_halfplane)
        if loss.item() < 1e-4: break
        loss.backward()
        opt_hp.step()
        
    print(f"  Halfplane Induction Complete in: {time.perf_counter() - t0:.4f}s")
    print(f"  Induced Symbolic Horn Clause: halfplane(X, Y) :- leq(add(mult({model_hp.a.item():.3f}, X), Y), {model_hp.b.item():.3f}).")

if __name__ == "__main__":
    execute_geometry_induction()
