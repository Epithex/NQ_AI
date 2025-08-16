#!/usr/bin/env python3
"""
Simple Visual ViT Model for NQ Pattern Analysis
Pure TensorFlow/Keras implementation without transformers dependency
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import logging

class PatchEmbedding(tf.keras.layers.Layer):
    """Image to patch embedding layer."""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16, embed_dim: int = 768, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch extraction via convolution
        self.projection = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='patch_projection'
        )
        
    def call(self, x):
        # Extract patches: (batch_size, num_patches, embed_dim)
        x = self.projection(x)  # (batch_size, grid_height, grid_width, embed_dim)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, self.num_patches, self.embed_dim])
        return x

class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head self-attention layer."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = tf.keras.layers.Dense(embed_dim * 3, use_bias=False)
        self.attn_drop = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(embed_dim)
        self.proj_drop = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, training=None):
        batch_size, seq_len, embed_dim = tf.unstack(tf.shape(x))
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, 3 * embed_dim)
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # (3, batch_size, num_heads, seq_len, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = tf.cast(self.head_dim, tf.float32) ** -0.5
        attn = tf.matmul(q, k, transpose_b=True) * scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        
        # Apply attention to values
        x = tf.matmul(attn, v)  # (batch_size, num_heads, seq_len, head_dim)
        x = tf.transpose(x, [0, 2, 1, 3])  # (batch_size, seq_len, num_heads, head_dim)
        x = tf.reshape(x, [batch_size, seq_len, embed_dim])
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        
        return x

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_hidden_dim, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout)
        ])
        
    def call(self, x, training=None):
        # Attention with residual connection
        x = x + self.attn(self.norm1(x), training=training)
        
        # MLP with residual connection  
        x = x + self.mlp(self.norm2(x), training=training)
        
        return x

class SimpleViT(tf.keras.Model):
    """Simple Vision Transformer for 4-class pattern classification."""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 4,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        
        # Class token and position embeddings
        self.cls_token = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer='zeros',
            trainable=True,
            name='cls_token'
        )
        
        self.pos_embed = self.add_weight(
            shape=(1, self.num_patches + 1, embed_dim),
            initializer='zeros',
            trainable=True,
            name='pos_embed'
        )
        
        self.pos_drop = tf.keras.layers.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ]
        
        # Classification head
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.head = tf.keras.layers.Dense(num_classes, activation='softmax')
        
        # Model will be built later
    
    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.embed_dim])
        x = tf.concat([cls_tokens, x], axis=1)  # (batch_size, num_patches + 1, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Classification head (use CLS token)
        x = self.norm(x)
        cls_output = x[:, 0]  # Extract CLS token
        x = self.head(cls_output)
        
        return x

def create_simple_visual_model(
    image_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 4,
    embed_dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    dropout: float = 0.1
) -> SimpleViT:
    """Create a simple visual ViT model."""
    
    model = SimpleViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )
    
    # Build the model
    model.build((None, image_size, image_size, 3))
    
    logging.info("Simple Visual ViT Model Architecture:")
    logging.info(f"  - Input: ({image_size}, {image_size}, 3) images")
    logging.info(f"  - Patches: {patch_size}x{patch_size}, {model.num_patches} total")
    logging.info(f"  - Embedding: {embed_dim} dimensions")
    logging.info(f"  - Layers: {num_layers} transformer blocks")
    logging.info(f"  - Heads: {num_heads} attention heads")
    logging.info(f"  - Classes: {num_classes}")
    try:
        param_count = model.count_params()
        logging.info(f"  - Total parameters: {param_count:,}")
    except:
        logging.info(f"  - Parameters will be counted after first forward pass")
    
    return model

def compile_model(
    model: SimpleViT,
    learning_rate: float = 3e-4,
    class_weights: Optional[dict] = None
) -> SimpleViT:
    """Compile model with optimizer and loss."""
    
    # Learning rate schedule
    learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=1000,
        t_mul=1.0,
        m_mul=1.0,
        alpha=0.1
    )
    
    # Optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=0.01
    )
    
    # Loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    # Metrics
    metrics = [
        'accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    logging.info("Model compiled successfully:")
    logging.info(f"  - Optimizer: AdamW (lr={learning_rate}, wd=0.01)")
    logging.info(f"  - Loss: SparseCategoricalCrossentropy")
    logging.info(f"  - Metrics: accuracy, top_2_accuracy")
    
    return model

if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)
    
    print("Creating Simple Visual ViT model...")
    model = create_simple_visual_model()
    
    print("\nCompiling model...")
    model = compile_model(model)
    
    print("\nModel ready for training!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Test forward pass
    test_input = tf.random.normal((2, 224, 224, 3))
    output = model(test_input)
    print(f"Test output shape: {output.shape}")
    print("Model test complete!")