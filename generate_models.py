"""
Script to generate ONNX models from PyTorch models for production use
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.edge_inference import convert_to_onnx

def generate_lstm_model():
    """Generate ONNX model for LSTM predictor"""
    print("Generating LSTM ONNX model...")
    
    try:
        from models.lstm_predictor.attention_lstm import create_enhanced_lstm
        
        # Create model with appropriate parameters
        model = create_enhanced_lstm(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=6,  # 6 defect types
            use_quantile=False,
            dropout=0.2
        )
        
        # Convert to ONNX
        convert_to_onnx(
            model,
            input_shape=(1, 30, 20),  # batch_size=1, seq_len=30, input_size=20
            output_path="models/lstm_predictor/lstm_model.onnx",
            opset_version=12  # Use higher opset version for einsum support
        )
        
        print("✅ LSTM ONNX model generated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate LSTM ONNX model: {e}")
        return False

def generate_vit_model():
    """Generate ONNX model for Vision Transformer"""
    print("Generating Vision Transformer ONNX model...")
    
    try:
        from models.vision_transformer.defect_detector import create_vit_classifier
        
        # Create model
        model = create_vit_classifier(
            img_size=32,
            patch_size=8,
            in_channels=3,
            n_classes=6,  # 6 defect types
            embed_dim=128,
            depth=6,
            n_heads=8
        )
        
        # Convert to ONNX
        convert_to_onnx(
            model,
            input_shape=(1, 3, 32, 32),  # batch_size=1, channels=3, height=32, width=32
            output_path="models/vision_transformer/vit_model.onnx",
            opset_version=12  # Use higher opset version for einsum support
        )
        
        print("✅ Vision Transformer ONNX model generated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate Vision Transformer ONNX model: {e}")
        return False

def generate_gnn_model():
    """Generate ONNX model for GNN using the actual GNN architecture"""
    print("Generating GNN ONNX model...")
    
    try:
        from models.gnn_sensor_network.gnn_model import create_dynamic_gnn
        import torch.nn as nn
        
        # Create model using the actual factory function
        model = create_dynamic_gnn(
            num_sensors=20,  # Match the expected input size
            input_dim=10,    # Match the expected input size
            hidden_dim=64,
            output_dim=32,
            edge_dim=3
        )
        
        # Create a wrapper to handle the GNN's specific input requirements
        class GNNWrapper(nn.Module):
            def __init__(self, gnn_model):
                super().__init__()
                self.gnn = gnn_model
                # Add a final classification layer to match expected output size
                self.classifier = nn.Linear(32, 6)  # 6 defect types
            
            def forward(self, node_features):
                # Create a simple edge index (connecting each node to the next one in a ring)
                num_nodes = node_features.shape[0]
                edge_index = torch.zeros(2, num_nodes, dtype=torch.long)
                for i in range(num_nodes):
                    edge_index[0, i] = i
                    edge_index[1, i] = (i + 1) % num_nodes
                
                # Create simple edge attributes
                edge_attr = torch.ones(num_nodes, 3, dtype=torch.float)
                
                # Run through GNN
                gnn_output, _ = self.gnn(node_features, edge_index, edge_attr, return_attention=False)
                
                # Global pooling (mean)
                pooled_output = gnn_output.mean(dim=0, keepdim=True)
                
                # Classification
                output = self.classifier(pooled_output)
                return output
        
        wrapper_model = GNNWrapper(model)
        
        # Convert to ONNX
        convert_to_onnx(
            wrapper_model,
            input_shape=(20, 10),  # num_nodes=20, node_features=10
            output_path="models/gnn_sensor_network/gnn_model.onnx",
            opset_version=16  # Use higher opset version for scatter_reduce support
        )
        
        print("✅ GNN ONNX model generated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate GNN ONNX model: {e}")
        return False

def main():
    """Generate all ONNX models"""
    print("🚀 Starting ONNX model generation for production...")
    
    success_count = 0
    total_count = 3
    
    # Generate LSTM model
    if generate_lstm_model():
        success_count += 1
    
    # Generate Vision Transformer model
    if generate_vit_model():
        success_count += 1
    
    # Generate GNN model
    if generate_gnn_model():
        success_count += 1
    
    print(f"\n📊 Generation Summary: {success_count}/{total_count} models generated successfully")
    
    if success_count == total_count:
        print("✅ All ONNX models generated successfully!")
        return True
    else:
        print("⚠️ Some models failed to generate. Check logs above.")
        return False

if __name__ == "__main__":
    main()