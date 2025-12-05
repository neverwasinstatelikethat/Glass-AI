"""
Test script to verify the GNN dataset generation fix
"""
import logging
import sys
import os

# Add the training directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

from training.synthetic_training_data_generator import SyntheticTrainingDataGenerator, DatasetConfig

def test_gnn_generation():
    """Test GNN dataset generation with small dataset"""
    logging.basicConfig(level=logging.INFO)
    
    generator = SyntheticTrainingDataGenerator(seed=42)
    
    # Generate small GNN dataset for testing
    print("🔄 Generating small GNN test dataset...")
    gnn_config = DatasetConfig(
        num_samples=100,  # Much smaller for testing
        positive_ratio=0.15,
        feature_dim=10  # Temporal statistics per sensor
    )
    gnn_dataset = generator.generate_gnn_dataset(gnn_config)
    
    print(f"✅ Generated GNN dataset with {len(gnn_dataset[0])} graphs")
    
    # Test saving the dataset
    print("💾 Saving GNN test dataset...")
    generator.save_dataset(gnn_dataset, "training/datasets/gnn_test_data.npz", 'gnn')
    
    gnn_stats = generator.get_dataset_statistics(gnn_dataset, 'gnn')
    print(f"📊 GNN dataset stats: {gnn_stats}")
    
    print("✅ GNN test completed successfully")

if __name__ == "__main__":
    test_gnn_generation()