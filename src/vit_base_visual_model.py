#!/usr/bin/env python3
"""
Pure Visual ViT-Base Model for NQ Pattern Analysis
Image-only input with no numerical features
"""

import tensorflow as tf
from transformers import TFViTModel, ViTConfig
import numpy as np
from typing import Tuple, Optional
import logging

class PureVisualViTBase(tf.keras.Model):
    """Pure visual ViT-Base model for 4-class pattern classification."""
    
    def __init__(
        self,
        num_classes: int = 4,
        image_size: int = 224,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        
        # ViT-Base configuration
        self.vit_config = ViTConfig(
            image_size=image_size,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            num_labels=num_classes
        )
        
        # Load pre-trained ViT-Base
        self.vit_base = TFViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            config=self.vit_config,
            from_tf=False
        )
        
        # Classification head (simple and effective)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(
                512,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                name='classifier_dense'
            ),
            tf.keras.layers.Dropout(dropout_rate, name='classifier_dropout'),
            tf.keras.layers.Dense(
                num_classes,
                activation='softmax',
                name='classifier_output'
            )
        ])
        
        # Build the model
        self.build((None, image_size, image_size, 3))
        
        logging.info(f"Created PureVisualViTBase with {self.count_params():,} parameters")
    
    def call(self, inputs, training=None):
        """Forward pass with image input only."""
        # Ensure inputs are in correct format
        if len(inputs.shape) == 4:
            images = inputs
        else:
            raise ValueError(f"Expected 4D image tensor, got shape: {inputs.shape}")
        
        # ViT expects pixel values in range [0, 1]
        # Convert from [0, 255] if needed
        if tf.reduce_max(images) > 1.0:
            images = images / 255.0
        
        # ViT forward pass
        vit_outputs = self.vit_base(pixel_values=images, training=training)
        
        # Get [CLS] token representation
        cls_token = vit_outputs.last_hidden_state[:, 0]  # Shape: (batch_size, 768)
        
        # Classification
        logits = self.classifier(cls_token, training=training)
        
        return logits
    
    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)

def create_pure_visual_model(
    num_classes: int = 4,
    image_size: int = 224,
    dropout_rate: float = 0.3
) -> PureVisualViTBase:
    """Create a pure visual ViT-Base model."""
    
    model = PureVisualViTBase(
        num_classes=num_classes,
        image_size=image_size,
        dropout_rate=dropout_rate
    )
    
    # Log model summary
    logging.info("Pure Visual ViT-Base Model Architecture:")
    logging.info(f"  - Input: ({image_size}, {image_size}, 3) images only")
    logging.info(f"  - ViT-Base: 12 layers, 768 hidden, 12 heads")
    logging.info(f"  - Classification: 512 → {num_classes} classes")
    logging.info(f"  - Dropout: {dropout_rate}")
    logging.info(f"  - Total parameters: {model.count_params():,}")
    
    return model

def compile_model(
    model: PureVisualViTBase,
    learning_rate: float = 3e-4,
    class_weights: Optional[dict] = None
) -> PureVisualViTBase:
    """Compile model with optimizer and loss."""
    
    # AdamW optimizer with cosine decay
    total_steps = 1000  # Will be updated during training
    warmup_steps = int(0.1 * total_steps)
    
    learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=total_steps,
        t_mul=1.0,
        m_mul=1.0,
        alpha=0.1
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=0.01,
        clipnorm=1.0
    )
    
    # Loss function with class weights if provided
    if class_weights:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    # Metrics
    metrics = [
        'accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
        tf.keras.metrics.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy')
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
    
    print("Creating Pure Visual ViT-Base model...")
    model = create_pure_visual_model()
    
    print("\nCompiling model...")
    model = compile_model(model)
    
    print("\nModel ready for training!")
    print(f"Total parameters: {model.count_params():,}")