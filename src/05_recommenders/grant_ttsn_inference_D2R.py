"""
Inference script for Grant Matching Two-Tower Model

Usage:
    python inference.py --donor_ein 911583492 --top_k 10

Or use interactively in Python:
    from inference import load_model, get_recommendations
    model, mappings = load_model()
    recs = get_recommendations(911583492, model, mappings)
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
# Get Recommendations
# =============================================================================

def get_recommendations(
        donor_ein: int,
        model: SimpleTwoTowerModel,
        mappings: Dict,
        top_k: int = 10,
        device: str = 'cpu',
        batch_size: int = 10000
) -> List[Tuple[int, float]]:
    """
    Get top K receiver recommendations for a donor EIN.

    Args:
        donor_ein: Donor EIN number
        model: Trained model
        mappings: Dictionary with EIN<->ID mappings
        top_k: Number of recommendations to return
        device: Device to run inference on
        batch_size: Batch size for scoring (for speed)

    Returns:
        List of (receiver_ein, score) tuples, sorted by score descending
    """
    donor_ein_to_id = mappings['donor_ein_to_id']
    receiver_id_to_ein = mappings['receiver_id_to_ein']

    # Check if donor exists
    if donor_ein not in donor_ein_to_id:
        raise ValueError(f"Donor EIN {donor_ein} not found in training data")

    donor_id = donor_ein_to_id[donor_ein]
    all_receiver_ids = list(receiver_id_to_ein.keys())

    print(f"Computing scores for {len(all_receiver_ids):,} receivers...")
    start_time = time.time()

    with torch.no_grad():
        # Get donor embedding once
        donor_emb = model.get_donor_embedding(donor_id).to(device)

        # Score all receivers in batches (for speed)
        scores = []
        for i in range(0, len(all_receiver_ids), batch_size):
            batch_receiver_ids = all_receiver_ids[i:i + batch_size]

            # Get batch of receiver embeddings
            receiver_ids_tensor = torch.tensor(batch_receiver_ids, dtype=torch.long).to(device)
            receiver_embs = model.receiver_embedding(receiver_ids_tensor)

            # Compute scores (dot product)
            batch_scores = (donor_emb * receiver_embs).sum(dim=1)

            # Store with EINs
            for receiver_id, score in zip(batch_receiver_ids, batch_scores):
                receiver_ein = receiver_id_to_ein[receiver_id]
                scores.append((receiver_ein, score.item()))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

    elapsed = time.time() - start_time
    print(f"✓ Scored {len(all_receiver_ids):,} receivers in {elapsed:.2f}s\n")

    return scores[:top_k]


def get_batch_recommendations(
        donor_eins: List[int],
        model: SimpleTwoTowerModel,
        mappings: Dict,
        top_k: int = 10,
        device: str = 'cpu'
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Get recommendations for multiple donors at once.

    Args:
        donor_eins: List of donor EIN numbers
        model: Trained model
        mappings: Dictionary with EIN<->ID mappings
        top_k: Number of recommendations per donor
        device: Device to run inference on

    Returns:
        Dictionary mapping donor_ein → list of (receiver_ein, score) tuples
    """
    results = {}

    for donor_ein in donor_eins:
        try:
            recs = get_recommendations(donor_ein, model, mappings, top_k, device)
            results[donor_ein] = recs
        except ValueError as e:
            print(f"Warning: {e}")
            results[donor_ein] = []

    return results


# =============================================================================
# Display Functions
# =============================================================================

def display_recommendations(donor_ein: int, recommendations: List[Tuple[int, float]]):
    """Pretty print recommendations."""
    print("=" * 70)
    print(f"Top {len(recommendations)} Recommendations for Donor EIN {donor_ein}")
    print("=" * 70)

    for rank, (receiver_ein, score) in enumerate(recommendations, 1):
        print(f"{rank:2d}. Receiver EIN {receiver_ein:>12} | Score: {score:7.4f}")

    print("=" * 70)


def analyze_donor(donor_ein: int, model: SimpleTwoTowerModel, mappings: Dict):
    """Analyze a donor's embedding and characteristics."""
    donor_ein_to_id = mappings['donor_ein_to_id']
    donor_id_to_ein = mappings['donor_id_to_ein']

    if donor_ein not in donor_ein_to_id:
        print(f"Error: Donor EIN {donor_ein} not found")
        return

    donor_id = donor_ein_to_id[donor_ein]

    with torch.no_grad():
        donor_emb = model.get_donor_embedding(donor_id)

        print(f"\nDonor EIN: {donor_ein}")
        print(f"Donor ID: {donor_id}")
        print(f"Embedding shape: {donor_emb.shape}")
        print(f"Embedding mean: {donor_emb.mean().item():.4f}")
        print(f"Embedding std: {donor_emb.std().item():.4f}")
        print(f"Embedding norm: {torch.norm(donor_emb).item():.4f}")
        print(f"First 10 dimensions: {donor_emb[0][:10].tolist()}")

        # Find most similar donors
        print(f"\nFinding most similar donors...")
        similarities = []
        for other_ein, other_id in list(donor_ein_to_id.items())[:1000]:  # Sample 1000 for speed
            if other_id == donor_id:
                continue
            other_emb = model.get_donor_embedding(other_id)
            sim = (donor_emb * other_emb).sum().item()
            similarities.append((other_ein, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 5 most similar donors:")
        for rank, (similar_ein, sim) in enumerate(similarities[:5], 1):
            print(f"  {rank}. EIN {similar_ein}: similarity = {sim:.4f}")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Get grant recommendations')
    parser.add_argument('--donor_ein', type=int, required=True, help='Donor EIN number')
    parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Path to model file')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--analyze', action='store_true', help='Analyze donor embedding')

    args = parser.parse_args()

    # Load model
    model, mappings = load_model(
        model_path=args.model_path,
        embedding_dim=args.embedding_dim,
        device=args.device
    )

    # Get recommendations
    try:
        recommendations = get_recommendations(
            donor_ein=args.donor_ein,
            model=model,
            mappings=mappings,
            top_k=args.top_k,
            device=args.device
        )

        display_recommendations(args.donor_ein, recommendations)

        # Optionally analyze donor
        if args.analyze:
            analyze_donor(args.donor_ein, model, mappings)

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
        print("Grant Matching Inference - Interactive Example")
        print("=" * 70)
        print("\nLoading model...")

        model, mappings = load_model()

        # Example usage
        donor_ein = 911583492

        print(f"\nGetting recommendations for donor EIN {donor_ein}...")
        recommendations = get_recommendations(
            donor_ein=donor_ein,
            model=model,
            mappings=mappings,
            top_k=10
        )

        display_recommendations(donor_ein, recommendations)

        print("\n" + "=" * 70)
        print("Usage Examples:")
        print("=" * 70)
        print("\n# Command line:")
        print("python inference.py --donor_ein 911583492 --top_k 20")
        print("\n# In Python script:")
        print("from inference import load_model, get_recommendations")
        print("model, mappings = load_model()")
        print("recs = get_recommendations(911583492, model, mappings, top_k=10)")
        print("\n# Analyze donor:")
        print("python inference.py --donor_ein 911583492 --analyze")
        print("=" * 70)