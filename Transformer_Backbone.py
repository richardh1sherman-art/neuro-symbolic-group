import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Ingest the exact Custom Dataset class and Collate function we just verified
from hanoi_dataset_loader import UniversalGroupDataset, universal_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfSimilarPolicySLM(nn.Module):
    """
    A custom Small Language Model built from scratch.
    Learns to project continuous thought vectors over self-similar group spaces.
    """
    def __init__(self, state_vocab_size, op_vocab_size, embedding_dim=64):
        super().__init__()
        # 1. Learnable Embedding Layer: Maps discrete token indices to 64D continuous thought space
        self.state_embedding = nn.Embedding(state_vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Continuous Thought Reasoning Layers: Blends input sequence state profiles
        self.reasoning_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
        
        # 3. Policy Token Projection Head: Maps continuous thought down to our 3 operator choices
        self.policy_head = nn.Linear(embedding_dim, op_vocab_size)

    def forward(self, input_state_tokens):
        # Transform discrete tokens to continuous embedding vectors
        embedded_states = self.state_embedding(input_state_tokens) # Shape: [Batch, Seq_Len, 64]
        
        # Pool the sequence context along the time dimension (temporal mean)
        continuous_thought = torch.mean(embedded_states, dim=1) # Shape: [Batch, 64]
        
        # Run through the reasoning layers
        processed_thought = self.reasoning_layer(continuous_thought)
        
        # Output continuous choice logits
        logits = self.policy_head(processed_thought) # Shape: [Batch, Op_Vocab_Size]
        return logits

# --- MAIN MODEL TRAINING WORKLOAD ---
if __name__ == "__main__":
    print("=== INITIALIZING CUSTOM SLM TRAINING WORKLOAD ===")
    print(f"Hardware Backbone Core: {device}\n")
    
    # 1. Load your verified dataset corpus
    json_path = "/home/rsherman/projects/SMT-ILP/universal_group_dataset.json"
    dataset = UniversalGroupDataset(json_path)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=universal_collate_fn)
    
    # 2. Instantiate your custom SLM
    state_v_size = len(dataset.state_vocab) # 9 tokens
    op_v_size = len(dataset.op_vocab)       # 3 tokens
    
    slm_model = SelfSimilarPolicySLM(state_vocab_size=state_v_size, op_vocab_size=op_v_size).to(device)
    
    # Setup optimization parameters
    optimizer = optim.Adam(slm_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding tokens during loss calc
    
    print(f"Custom SLM Compiled successfully from scratch.")
    print(f"Model Parameters: State Vocab={state_v_size} | Operator Vocab={op_v_size} | Embedding Dim=64")
    print("-" * 75)
    
    # 3. Execute 5 local training iterations
    slm_model.train()
    for epoch in range(1, 6):
        total_loss = 0
        for inputs, targets, outputs in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Project continuous logits from our embeddings
            predicted_logits = slm_model(inputs)
            
            # For cross-entropy loss matching, compress sequence dimensions 
            # We align against the main target choices (index 0 step targets)
            loss = criterion(predicted_logits, targets[:, 0])
            
            # Backpropagation pass to update your custom weights
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}/5 | Optimization Loss Penalty: {total_loss:.6f}")
        
    print("-" * 75)
    print("✅ TRAINING ITERATION COMPLETE: Weights successfully optimized locally.")
