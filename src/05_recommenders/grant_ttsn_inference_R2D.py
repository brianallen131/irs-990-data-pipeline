"""
Reverse Inference script for Grant Matching Two-Tower Model
Find potential donors for a given nonprofit receiver

Usage:
    python reverse_inference.py --receiver_ein 123456789 --top_k 20

Or use interactively in Python:
    from reverse_inference import load_model, get_potential_donors
    model, mappings = load_model()
    donors = get_potential_donors(123456789, model, mappings)
"""

import torch
import pickle
import argparse
from typing import List, Tuple, Dict
import time


# =============================================================================
# Model Definition (must match training)
# =============================================================================

class SimpleTwoTowerModel(torch.nn.Module):
    """Simple two-tower model with just ID embeddings."""

    def __init__(self, num_donors: int, num_receivers: int, embedding_dim: int = 128):
        super().__init__()
        self.donor_embedding = torch.nn.Embedding(num_donors, embedding_dim)
        self.receiver_embedding = torch.nn.Embedding(num_receivers, embedding_dim)

    def forward(self, donor_ids: torch.Tensor, receiver_ids: torch.Tensor) -> torch.Tensor:
        donor_emb = self.donor_embedding(donor_ids)
        receiver_emb = self.receiver_embedding(receiver_ids)
        scores = (donor_emb * receiver_emb).sum(dim=1)
        return scores

    def get_donor_embedding(self, donor_id: int) -> torch.Tensor:
        donor_id = torch.tensor([donor_id], dtype=torch.long)
        return self.donor_embedding(donor_id)

    def get_receiver_embedding(self, receiver_id: int) -> torch.Tensor:
        receiver_id = torch.tensor([receiver_id], dtype=torch.long)
        return self.receiver_embedding(receiver_id)


# =============================================================================
# Load Model and Mappings
# =============================================================================

def load_model(
        model_path: str = 'best_model.pt',
        donor_ein_to_id_path: str = 'donor_ein_to_id.pkl',
        receiver_id_to_ein_path: str = 'receiver_id_to_ein.pkl',
        donor_id_to_ein_path: str = 'donor_id_to_ein.pkl',
        receiver_ein_to_id_path: str = 'receiver_ein_to_id.pkl',
        embedding_dim: int = 128,
        device: str = 'cpu'
) -> Tuple[SimpleTwoTowerModel, Dict]:
    """
    Load trained model and EIN mappings.

    Returns:
        model: Loaded PyTorch model
        mappings: Dictionary with all EIN<->ID mappings
    """
    print("Loading mappings...")

    # Load mappings
    with open(donor_ein_to_id_path, 'rb') as f:
        donor_ein_to_id = pickle.load(f)
    with open(receiver_id_to_ein_path, 'rb') as f:
        receiver_id_to_ein = pickle.load(f)
    with open(donor_id_to_ein_path, 'rb') as f:
        donor_id_to_ein = pickle.load(f)
    with open(receiver_ein_to_id_path, 'rb') as f:
        receiver_ein_to_id = pickle.load(f)

    num_donors = len(donor_ein_to_id)
    num_receivers = len(receiver_ein_to_id)

    print(f"  Unique donors: {num_donors:,}")
    print(f"  Unique receivers: {num_receivers:,}")

    # Initialize model
    print(f"\nInitializing model (embedding_dim={embedding_dim})...")
    model = SimpleTwoTowerModel(
        num_donors=num_donors,
        num_receivers=num_receivers,
        embedding_dim=embedding_dim
    )

    # Load weights
    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully!\n")

    mappings = {
        'donor_ein_to_id': donor_ein_to_id,
        'receiver_id_to_ein': receiver_id_to_ein,
        'donor_id_to_ein': donor_id_to_ein,
        'receiver_ein_to_id': receiver_ein_to_id
    }

    return model, mappings


# =============================================================================
# Get Potential Donors for a Receiver
# =============================================================================

def get_potential_donors(
        receiver_ein: int,
        model: SimpleTwoTowerModel,
        mappings: Dict,
        top_k: int = 20,
        device: str = 'cpu',
        batch_size: int = 10000
) -> List[Tuple[int, float]]:
    """
    Get top K potential donors for a receiver (nonprofit).

    Args:
        receiver_ein: Receiver EIN number
        model: Trained model
        mappings: Dictionary with EIN<->ID mappings
        top_k: Number of donors to return
        device: Device to run inference on
        batch_size: Batch size for scoring (for speed)

    Returns:
        List of (donor_ein, score) tuples, sorted by score descending
    """
    receiver_ein_to_id = mappings['receiver_ein_to_id']
    donor_id_to_ein = mappings['donor_id_to_ein']

    # Check if receiver exists
    if receiver_ein not in receiver_ein_to_id:
        raise ValueError(f"Receiver EIN {receiver_ein} not found in training data")

    receiver_id = receiver_ein_to_id[receiver_ein]
    all_donor_ids = list(donor_id_to_ein.keys())

    print(f"Computing scores for {len(all_donor_ids):,} donors...")
    start_time = time.time()

    with torch.no_grad():
        # Get receiver embedding once
        receiver_emb = model.get_receiver_embedding(receiver_id).to(device)

        # Score all donors in batches (for speed)
        scores = []
        for i in range(0, len(all_donor_ids), batch_size):
            batch_donor_ids = all_donor_ids[i:i + batch_size]

            # Get batch of donor embeddings
            donor_ids_tensor = torch.tensor(batch_donor_ids, dtype=torch.long).to(device)
            donor_embs = model.donor_embedding(donor_ids_tensor)

            # Compute scores (dot product)
            batch_scores = (receiver_emb * donor_embs).sum(dim=1)

            # Store with EINs
            for donor_id, score in zip(batch_donor_ids, batch_scores):
                donor_ein = donor_id_to_ein[donor_id]
                scores.append((donor_ein, score.item()))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

    elapsed = time.time() - start_time
    print(f"✓ Scored {len(all_donor_ids):,} donors in {elapsed:.2f}s\n")

    return scores[:top_k]


def get_batch_potential_donors(
        receiver_eins: List[int],
        model: SimpleTwoTowerModel,
        mappings: Dict,
        top_k: int = 20,
        device: str = 'cpu'
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Get potential donors for multiple receivers at once.

    Args:
        receiver_eins: List of receiver EIN numbers
        model: Trained model
        mappings: Dictionary with EIN<->ID mappings
        top_k: Number of donors per receiver
        device: Device to run inference on

    Returns:
        Dictionary mapping receiver_ein → list of (donor_ein, score) tuples
    """
    results = {}

    for receiver_ein in receiver_eins:
        try:
            donors = get_potential_donors(receiver_ein, model, mappings, top_k, device)
            results[receiver_ein] = donors
        except ValueError as e:
            print(f"Warning: {e}")
            results[receiver_ein] = []

    return results


# =============================================================================
# Display Functions
# =============================================================================

def display_potential_donors(receiver_ein: int, donors: List[Tuple[int, float]]):
    """Pretty print potential donors."""
    print("=" * 70)
    print(f"Top {len(donors)} Potential Donors for Receiver EIN {receiver_ein}")
    print("=" * 70)

    for rank, (donor_ein, score) in enumerate(donors, 1):
        print(f"{rank:2d}. Donor EIN {donor_ein:>12} | Score: {score:7.4f}")

    print("=" * 70)


def analyze_receiver(receiver_ein: int, model: SimpleTwoTowerModel, mappings: Dict):
    """Analyze a receiver's embedding and characteristics."""
    receiver_ein_to_id = mappings['receiver_ein_to_id']

    if receiver_ein not in receiver_ein_to_id:
        print(f"Error: Receiver EIN {receiver_ein} not found")
        return

    receiver_id = receiver_ein_to_id[receiver_ein]

    with torch.no_grad():
        receiver_emb = model.get_receiver_embedding(receiver_id)

        print(f"\nReceiver EIN: {receiver_ein}")
        print(f"Receiver ID: {receiver_id}")
        print(f"Embedding shape: {receiver_emb.shape}")
        print(f"Embedding mean: {receiver_emb.mean().item():.4f}")
        print(f"Embedding std: {receiver_emb.std().item():.4f}")
        print(f"Embedding norm: {torch.norm(receiver_emb).item():.4f}")
        print(f"First 10 dimensions: {receiver_emb[0][:10].tolist()}")

        # Find most similar receivers
        print(f"\nFinding most similar receivers...")
        receiver_id_to_ein = mappings['receiver_id_to_ein']
        similarities = []

        # Sample for speed
        sample_receivers = list(receiver_ein_to_id.items())[:1000]
        for other_ein, other_id in sample_receivers:
            if other_id == receiver_id:
                continue
            other_emb = model.get_receiver_embedding(other_id)
            sim = (receiver_emb * other_emb).sum().item()
            similarities.append((other_ein, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 5 most similar receivers:")
        for rank, (similar_ein, sim) in enumerate(similarities[:5], 1):
            print(f"  {rank}. EIN {similar_ein}: similarity = {sim:.4f}")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Get potential donors for a nonprofit')
    parser.add_argument('--receiver_ein', type=int, required=True, help='Receiver (nonprofit) EIN number')
    parser.add_argument('--top_k', type=int, default=20, help='Number of potential donors')
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Path to model file')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--analyze', action='store_true', help='Analyze receiver embedding')

    args = parser.parse_args()

    # Load model
    model, mappings = load_model(
        model_path=args.model_path,
        embedding_dim=args.embedding_dim,
        device=args.device
    )

    # Get potential donors
    try:
        donors = get_potential_donors(
            receiver_ein=args.receiver_ein,
            model=model,
            mappings=mappings,
            top_k=args.top_k,
            device=args.device
        )

        display_potential_donors(args.receiver_ein, donors)

        # Optionally analyze receiver
        if args.analyze:
            analyze_receiver(args.receiver_ein, model, mappings)

    except ValueError as e:
        print(f"Error: {e}")


# =============================================================================
# Interactive Usage Example
# =============================================================================

if __name__ == "__main__":
    import sys

    # If run with arguments, use CLI
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive example
        print("=" * 70)
        print("Grant Matching Reverse Inference - Find Donors for Nonprofits")
        print("=" * 70)
        print("\nLoading model...")

        model, mappings = load_model()

        # Example usage - you'll need to replace with an actual receiver EIN
        print("\nExample: Finding potential donors for a receiver")
        print("To use: Provide a receiver EIN from your dataset")

        # Get a sample receiver EIN from the mappings
        sample_receiver_ein = list(mappings['receiver_ein_to_id'].keys())[0]

        print(f"\nGetting potential donors for receiver EIN {sample_receiver_ein}...")
        donors = get_potential_donors(
            receiver_ein=sample_receiver_ein,
            model=model,
            mappings=mappings,
            top_k=20
        )

        display_potential_donors(sample_receiver_ein, donors)

        print("\n" + "=" * 70)
        print("Usage Examples:")
        print("=" * 70)
        print("\n# Command line:")
        print("python reverse_inference.py --receiver_ein 123456789 --top_k 20")
        print("\n# In Python script:")
        print("from reverse_inference import load_model, get_potential_donors")
        print("model, mappings = load_model()")
        print("donors = get_potential_donors(123456789, model, mappings, top_k=20)")
        print("\n# Analyze receiver:")
        print("python reverse_inference.py --receiver_ein 123456789 --analyze")
        print("\n# Use specific epoch:")
        print("python reverse_inference.py --receiver_ein 123456789 --model_path model_epoch_10.pt --embedding_dim 64")
        print("=" * 70)