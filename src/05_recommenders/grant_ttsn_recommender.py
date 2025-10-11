from os import mkdir

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# =============================================================================
# MODEL
# =============================================================================

class SimpleTwoTowerModel(nn.Module):
    """
    Simple two-tower model: just learns embeddings for each donor and receiver ID.
    """

    def __init__(self, num_donors: int, num_receivers: int, embedding_dim: int = 128):
        super().__init__()
        self.donor_embedding = nn.Embedding(num_donors, embedding_dim)
        self.receiver_embedding = nn.Embedding(num_receivers, embedding_dim)

        # Initialize with small random values
        nn.init.normal_(self.donor_embedding.weight, std=0.01)
        nn.init.normal_(self.receiver_embedding.weight, std=0.01)

    def forward(self, donor_ids: torch.Tensor, receiver_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            donor_ids: [batch_size]
            receiver_ids: [batch_size]
        Returns:
            scores: [batch_size]
        """
        donor_emb = self.donor_embedding(donor_ids)  # [batch_size, 128]
        receiver_emb = self.receiver_embedding(receiver_ids)  # [batch_size, 128]

        # Dot product for similarity
        scores = (donor_emb * receiver_emb).sum(dim=1)  # [batch_size]
        return scores

    def get_donor_embedding(self, donor_id: int) -> torch.Tensor:
        """Get the embedding for a specific donor."""
        donor_id = torch.tensor([donor_id], dtype=torch.long)
        return self.donor_embedding(donor_id)

    def get_receiver_embedding(self, receiver_id: int) -> torch.Tensor:
        """Get the embedding for a specific receiver."""
        receiver_id = torch.tensor([receiver_id], dtype=torch.long)
        return self.receiver_embedding(receiver_id)


# =============================================================================
# DATASET
# =============================================================================

class GrantMatchingDataset(Dataset):
    """Dataset for training. Expects DataFrame with: donor_id, receiver_id, label"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'donor_id': int(row['donor_id']),
            'receiver_id': int(row['receiver_id']),
            'label': float(row['label'])  # Will be converted to float32 in dataloader
        }


def collate_fn(batch):
    """Custom collate function to ensure float32 for MPS compatibility."""
    return {
        'donor_id': torch.tensor([item['donor_id'] for item in batch], dtype=torch.long),
        'receiver_id': torch.tensor([item['receiver_id'] for item in batch], dtype=torch.long),
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.float32)  # float32 for MPS
    }


# =============================================================================
# TRAINING
# =============================================================================

def get_device():
    """Get the best available device (Apple Silicon GPU, NVIDIA GPU, or CPU)"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon GPU (M1/M2/M3/M4)
    else:
        return 'cpu'


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        patience: int = 5,
        device: str = None,
        force_device: bool = False
):
    """Train the two-tower model with early stopping."""
    if device is None or not force_device:
        device = get_device()

    print(f"Device requested: {device if device else 'auto'}")
    print(f"Using device: {device}")
    if device == 'mps':
        print("  → Apple Silicon GPU detected (M1/M2/M3/M4)")
    elif device == 'cuda':
        print(f"  → NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("  → Using CPU")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)  # No weight decay for now
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nTraining for up to {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()

            donor_ids = batch['donor_id'].to(device)
            receiver_ids = batch['receiver_id'].to(device)
            labels = batch['label'].to(device)

            scores = model(donor_ids, receiver_ids)
            loss = criterion(scores, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update progress bar with current loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]  ")
        with torch.no_grad():
            for batch in val_pbar:
                donor_ids = batch['donor_id'].to(device)
                receiver_ids = batch['receiver_id'].to(device)
                labels = batch['label'].to(device)

                scores = model(donor_ids, receiver_ids)
                loss = criterion(scores, labels)
                val_loss += loss.item()

                predictions = (torch.sigmoid(scores) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {accuracy:.4f}")

        # Save every 5th epoch
        if epoch in [0,4,9,14,19]:
            torch.save(model.state_dict(), f'models/model_epoch_{epoch + 1}.pt')
            print(f"  Model saved: model_epoch_{epoch + 1}.pt")

        # Also track best for reference
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/best_model.pt')
            print("  ✓ Best validation model saved: best_model.pt")

        print()  # Empty line between epochs

    # Save final model
    torch.save(model.state_dict(), 'models/final_model.pt')
    print(f"\n✓ Training complete!")
    print(f"  Final model saved: final_model.pt")
    print(f"  Best validation model saved: best_model.pt")
    print(f"  Every fifth epochs saved: model_epoch_1.pt through model_epoch_{num_epochs}.pt")

    return model


# =============================================================================
# INFERENCE
# =============================================================================

def find_top_matches(
        model: nn.Module,
        donor_id: int,
        all_receiver_ids: List[int],
        top_k: int = 10,
        device: str = None
) -> List[Tuple[int, float]]:
    """
    Find top K receiver matches for a given donor.

    Returns:
        List of (receiver_id, score) tuples, sorted by score descending
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        donor_emb = model.get_donor_embedding(donor_id).to(device)

        scores = []
        for receiver_id in all_receiver_ids:
            receiver_emb = model.get_receiver_embedding(receiver_id).to(device)
            score = (donor_emb * receiver_emb).sum().item()
            scores.append((receiver_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

if __name__ == "__main__":
    """
    Complete workflow for training SimpleTwoTowerModel on grant data.

    Your actual dimensions:
    - 163,678 grantors (donors)
    - 847,721 grantees (receivers)  
    - 4,125,248 grants
    """

    # Configuration
    EMBEDDING_DIM = 64  # Reduced from 128
    BATCH_SIZE = 1024
    NUM_EPOCHS = 20  # Train for  epochs
    LEARNING_RATE = 0.001
    NEGATIVE_RATIO = 5  # 2 negative samples per positive

    print("=" * 70)
    print("SIMPLE TWO-TOWER MODEL - GRANT MATCHING")
    print("=" * 70)
    print(f"Embedding dim: {EMBEDDING_DIM}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Negative ratio: {NEGATIVE_RATIO}")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Load your grant data
    # =========================================================================
    print("\n[STEP 1] Loading grant data...")

    df_gwe_dtypes = {
        'filing_number': float,
        'zip_name': str,
        'ein': float,
        'tax_yr': float,
        'form': str,
        'recipient_ein': float,
        'recipient_name': str,
        'recipient_address': str,
        'recipient_city': str,
        'recipient_state': str,
        'recipient_zip': str,
        'grant_purpose': str,
        'grant_amt': float,
        'clean_recipient_name': str,
        'clean_recipient_city': str,
        'clean_recipient_state': str,
        'clean_recipient_zip': str,
        'EIN': float,
        'match_score': float,
        'match_type': str
    }

    # Load your grant data
    df_gwe = pd.read_parquet('data/cleaned_irs_data/grant_details_recipients_inferred.parquet',
                             engine='fastparquet',
                             dtypes=df_gwe_dtypes
                             )
    df_gwe['receiver_ein'] = df_gwe['recipient_ein'].combine_first(df_gwe['EIN'])

    convert_to_int = ['filing_number', 'ein', 'tax_yr', 'recipient_ein', 'EIN', 'receiver_ein']
    for column in convert_to_int:
        df_gwe[column] = df_gwe[column].fillna(0).astype(int)

    df_gwe_input = df_gwe[['ein', 'receiver_ein', 'grant_amt']]
    df_gwe_input.columns = ['donor_ein', 'receiver_ein', 'grant_amt']

    mask = (df_gwe_input['grant_amt'] > 0) & (df_gwe_input['receiver_ein'] > 0)
    df = df_gwe_input[mask].groupby(['donor_ein', 'receiver_ein'])['grant_amt'].sum().sort_values(
        ascending=False).reset_index()

    print(f"Loaded {len(df):,} grants")
    print(f"Total grant amount: ${df['grant_amt'].sum():,.2f}")

    # =========================================================================
    # STEP 1b: Create ID mappings (EIN to integer index)
    # =========================================================================
    print("\n[STEP 1b] Creating EIN to ID mappings...")

    # Map EINs to integer IDs (0 to N-1) for embeddings
    unique_donor_eins = df['donor_ein'].unique()
    unique_receiver_eins = df['receiver_ein'].unique()

    donor_ein_to_id = {ein: idx for idx, ein in enumerate(unique_donor_eins)}
    receiver_ein_to_id = {ein: idx for idx, ein in enumerate(unique_receiver_eins)}

    # Reverse mappings (for converting back)
    donor_id_to_ein = {idx: ein for ein, idx in donor_ein_to_id.items()}
    receiver_id_to_ein = {idx: ein for ein, idx in receiver_ein_to_id.items()}

    # Apply mappings to create donor_id and receiver_id columns
    df['donor_id'] = df['donor_ein'].map(donor_ein_to_id)
    df['receiver_id'] = df['receiver_ein'].map(receiver_ein_to_id)

    # Update model dimensions based on actual unique entities
    NUM_DONORS = len(unique_donor_eins)
    NUM_RECEIVERS = len(unique_receiver_eins)

    print(f"Unique donors: {NUM_DONORS:,}")
    print(f"Unique receivers: {NUM_RECEIVERS:,}")

    # Save mappings for later use
    import pickle
    import os
    from pathlib import Path

    Path('models').mkdir(parents=True, exist_ok=True)

    with open('models/donor_ein_to_id.pkl', 'wb') as f:
        pickle.dump(donor_ein_to_id, f)
    with open('models/receiver_ein_to_id.pkl', 'wb') as f:
        pickle.dump(receiver_ein_to_id, f)
    with open('models/donor_id_to_ein.pkl', 'wb') as f:
        pickle.dump(donor_id_to_ein, f)
    with open('models/receiver_id_to_ein.pkl', 'wb') as f:
        pickle.dump(receiver_id_to_ein, f)

    print("Saved EIN mappings to .pkl files")

    # =========================================================================
    # STEP 2: Create training data with negatives
    # =========================================================================
    print(f"\n[STEP 2] Creating negative samples (ratio={NEGATIVE_RATIO})...")

    # Positive samples
    positives = df.copy()
    positives['label'] = 1

    # Negative samples
    actual_grants = set(zip(df['donor_id'], df['receiver_id']))
    negatives = []

    for donor_id in df['donor_id'].unique():
        num_donor_grants = len(df[df['donor_id'] == donor_id])
        sampled = 0
        attempts = 0
        max_attempts = num_donor_grants * NEGATIVE_RATIO * 3

        while sampled < num_donor_grants * NEGATIVE_RATIO and attempts < max_attempts:
            receiver_id = np.random.randint(0, NUM_RECEIVERS)
            if (donor_id, receiver_id) not in actual_grants:
                negatives.append({
                    'donor_id': donor_id,
                    'receiver_id': receiver_id,
                    'label': 0
                })
                sampled += 1
            attempts += 1

    negatives_df = pd.DataFrame(negatives)
    training_df = pd.concat([positives, negatives_df], ignore_index=True)
    training_df = training_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Positives: {len(positives):,}")
    print(f"Negatives: {len(negatives_df):,}")
    print(f"Total: {len(training_df):,}")

    # =========================================================================
    # STEP 3: Split train/validation
    # =========================================================================
    print("\n[STEP 3] Splitting train/validation (85/15)...")

    train_df, val_df = train_test_split(training_df, test_size=0.15, random_state=42)
    print(f"Train: {len(train_df):,}")
    print(f"Val: {len(val_df):,}")

    # =========================================================================
    # STEP 4: Create dataloaders
    # =========================================================================
    print(f"\n[STEP 4] Creating dataloaders (batch_size={BATCH_SIZE})...")

    train_dataset = GrantMatchingDataset(train_df)
    val_dataset = GrantMatchingDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # =========================================================================
    # STEP 5: Initialize model
    # =========================================================================
    print(f"\n[STEP 5] Initializing model...")

    model = SimpleTwoTowerModel(
        num_donors=NUM_DONORS,
        num_receivers=NUM_RECEIVERS,
        embedding_dim=EMBEDDING_DIM
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # =========================================================================
    # STEP 6: Train
    # =========================================================================
    print(f"\n[STEP 6] Training for up to {NUM_EPOCHS} epochs...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=5
    )

    # =========================================================================
    # STEP 7: Test inference
    # =========================================================================
    print("\n[STEP 7] Testing inference...")

    # Use actual donor_id from the mapped data (not the EIN)
    sample_donor_id = int(df['donor_id'].iloc[0])  # Ensure it's an int
    sample_donor_ein = df['donor_ein'].iloc[0]

    # Get candidate receivers - use actual receiver IDs from the data
    candidate_receivers = df['receiver_id'].unique()[:100].tolist()

    print(f"Sample donor ID: {sample_donor_id} (EIN: {sample_donor_ein})")
    print(f"Number of candidate receivers: {len(candidate_receivers)}")

    top_matches = find_top_matches(
        model,
        donor_id=sample_donor_id,
        all_receiver_ids=candidate_receivers,
        top_k=10,
        device=device
    )

    print(f"\nTop 10 matches for Donor EIN {sample_donor_ein} (ID={sample_donor_id}):")
    for i, (receiver_id, score) in enumerate(top_matches, 1):
        receiver_ein = receiver_id_to_ein[receiver_id]
        print(f"  {i}. Receiver EIN {receiver_ein} (ID={receiver_id}): {score:.4f}")

    print("\n" + "=" * 70)
    print("Model saved to: best_model.pt")
    print("EIN mappings saved to: *_ein_to_id.pkl files")
    print("=" * 70)
    print("\nTo use the model later:")
    print("1. Load mappings: donor_ein_to_id = pickle.load(open('donor_ein_to_id.pkl', 'rb'))")
    print("2. Convert EIN to ID: donor_id = donor_ein_to_id[ein]")
    print("3. Get recommendations: find_top_matches(model, donor_id, all_receiver_ids)")
    print("4. Convert IDs back to EINs using: receiver_id_to_ein[receiver_id]")
    print("=" * 70)