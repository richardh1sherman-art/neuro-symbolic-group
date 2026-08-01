import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class UniversalGroupDataset(Dataset):
    def __init__(self, json_path):
        # 1. Load the compiled neuro-symbolic JSON database
        with open(json_path, 'r') as f:
            self.raw_data = json.load(f)
            
        # 2. Build Vocabulary Dictionaries for token-to-index encoding
        # Reserve index 0 for the strict sequence padding token [PAD]
        self.state_vocab = {"[PAD]": 0}
        self.op_vocab = {"[PAD]": 0}
        
        # Build token lists dynamically from data properties
        for episode in self.raw_data:
            for s_tok in episode["token_sequences"]["input_state_tokens"]:
                if s_tok not in self.state_vocab:
                    self.state_vocab[s_tok] = len(self.state_vocab)
            
            # Ensure target destination state is registered in our vocabulary
            out_tok = episode["token_sequences"]["output_state_token"]
            if out_tok not in self.state_vocab:
                self.state_vocab[out_tok] = len(self.state_vocab)
                
            for o_tok in episode["token_sequences"]["target_operator_tokens"]:
                if o_tok not in self.op_vocab:
                    self.op_vocab[o_tok] = len(self.op_vocab)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        episode = self.raw_data[idx]
        seq_data = episode["token_sequences"]
        
        # 3. Map discrete token strings directly to integer indices
        input_states = [self.state_vocab[tok] for tok in seq_data["input_state_tokens"]]
        target_ops = [self.op_vocab[tok] for tok in seq_data["target_operator_tokens"]]
        output_state = self.state_vocab[seq_data["output_state_token"]]
        
        # Convert data blocks to tensor arrays
        return (
            torch.tensor(input_states, dtype=torch.long),
            torch.tensor(target_ops, dtype=torch.long),
            torch.tensor(output_state, dtype=torch.long)
        )

def universal_collate_fn(batch):
    """
    DYNAMIC BATCH PADDING: Packs varying multi-step token sequences 
    into unified tensor shapes to fully saturate GPU memory layout blocks.
    """
    input_states_batch, target_ops_batch, output_states_batch = zip(*batch)
    
    # Apply sequence padding (batch_first=True makes shape: [Batch, Sequence_Length])
    padded_inputs = pad_sequence(input_states_batch, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(target_ops_batch, batch_first=True, padding_value=0)
    
    # Output states are individual scalar elements, stack them directly
    output_states = torch.stack(output_states_batch)
    
    return padded_inputs, padded_targets, output_states

# --- RUN DATA LOADER VERIFICATION WORKLOAD ---
if __name__ == "__main__":
    print("=== INITIALIZING PYTORCH NEURO-SYMBOLIC DATA WORKLOAD ===")
    json_file_path = "/home/rsherman/projects/SMT-ILP/universal_group_dataset.json"
    
    # Initialize the custom dataset
    hanoi_dataset = UniversalGroupDataset(json_file_path)
    print(f"Dataset Vocabulary Sizes: State Vocab = {len(hanoi_dataset.state_vocab)} | Operator Vocab = {len(hanoi_dataset.op_vocab)}")
    
    # Instantiate the DataLoader utilizing our padding collate engine
    # Batch size set to 4 to verify simultaneous multi-episode packing
    data_loader = DataLoader(
        hanoi_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=universal_collate_fn
    )
    
    # Target your active environment hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting Processing Layer Core: {device}\n")
    
    # Pull a single optimized batch execution pass
    for batch_idx, (inputs, targets, outputs) in enumerate(data_loader):
        # Push variables straight onto the Blackwell Core
        inputs, targets, outputs = inputs.to(device), targets.to(device), outputs.to(device)
        
        print(f"Batch {batch_idx} Execution Shape Allocation:")
        print(f"  Padded Input States Shape  : {list(inputs.shape)} -> Tensor values packed on GPU")
        print(f"  Padded Target Operators Shape: {list(targets.shape)} -> Tensor values packed on GPU")
        print(f"  Destination Outputs Shape  : {list(outputs.shape)}")
        print("\nRaw Padded Inputs Content Tensor Layout:")
        print(inputs.cpu())
        print("\nRaw Padded Targets Content Tensor Layout:")
        print(targets.cpu())
        break  # Only run one batch sweep to verify parameters
