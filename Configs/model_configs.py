""" # Define your model configurations dictionary
MODEL_CONFIGS = {
    'model_base': {
        # Base model configuration
        'in_channels': 3,
        'image_size': 256,

        # Standard architecture parameters
        'embed_dim': 512,
        'depth': 12,
        'num_heads': 8,
        'decoder_embed_dim': 256,
        'decoder_depth': 4,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
        'activation': 'gelu',
        'recon_method': 'method1',
        'dropout': 0.05,

        # Additional parameters
        'segmentation_features': True,
        'edge_aware_loss': True
    }} """

MODEL_CONFIGS = {
    'model_tiny': {
        # Tiny model configuration
        'in_channels': 3,
        'image_size': 256,
        
        # Standard architecture parameters
        'embed_dim': 192,
        'depth': 8,
        'num_heads': 4,
        'decoder_embed_dim': 128,
        'decoder_depth': 2,
        'decoder_num_heads': 4,
        'mlp_ratio': 4,
        'activation': 'gelu',
        'recon_method': 'method1',
        'dropout': 0.1,
        
        # Additional parameters
        'segmentation_features': True,
        'edge_aware_loss': True
    },
    
    'model_small': {
        # Small model configuration
        'in_channels': 3,
        'image_size': 256,
        
        # Standard architecture parameters
        'embed_dim': 384,
        'depth': 10,
        'num_heads': 6,
        'decoder_embed_dim': 192,
        'decoder_depth': 3,
        'decoder_num_heads': 6,
        'mlp_ratio': 4,
        'activation': 'gelu',
        'recon_method': 'method1',
        'dropout': 0.07,
        
        # Additional parameters
        'segmentation_features': True,
        'edge_aware_loss': True
    },
    
    'model_base': {
        # Base model configuration
        'in_channels': 3,
        'image_size': 256,
        
        # Standard architecture parameters
        'embed_dim': 512,
        'depth': 12,
        'num_heads': 8,
        'decoder_embed_dim': 256,
        'decoder_depth': 4,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
        'activation': 'gelu',
        'recon_method': 'method1',
        'dropout': 0.05,
        
        # Additional parameters
        'segmentation_features': True,
        'edge_aware_loss': True
    },
    
    'model_large': {
        # Large model configuration
        'in_channels': 3,
        'image_size': 256,
        
        # Standard architecture parameters
        'embed_dim': 768,
        'depth': 16,
        'num_heads': 12,
        'decoder_embed_dim': 384,
        'decoder_depth': 6,
        'decoder_num_heads': 12,
        'mlp_ratio': 4,
        'activation': 'gelu',
        'recon_method': 'method1',
        'dropout': 0.03,
        
        # Additional parameters
        'segmentation_features': True,
        'edge_aware_loss': True
    },
    
    'model_huge': {
        # Huge model configuration
        'in_channels': 3,
        'image_size': 256,
        
        # Standard architecture parameters
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'mlp_ratio': 4,
        'activation': 'gelu',
        'recon_method': 'method1',
        'dropout': 0.02,
        
        # Additional parameters
        'segmentation_features': True,
        'edge_aware_loss': True
    }
}
