#!/usr/bin/env python3
"""
Proper ViT-Base Model for NQ Pattern Analysis
Using proven vit-keras implementation with ~87M parameters
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import logging
from vit_keras import vit

class ProperViTBase(tf.keras.Model):
    """Proper ViT-Base model using vit-keras with correct parameter count."""
    
    def __init__(
        self,
        image_size: int = 224,
        num_classes: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.image_size = image_size
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Create ViT-Base using vit-keras (this will have ~87M parameters)
        self.vit_base = vit.vit_b16(
            image_size=image_size,
            activation='softmax',
            pretrained=True,
            include_top=False,
            pretrained_top=False
        )
        
        # Freeze some layers initially (optional - can fine-tune all)
        # for layer in self.vit_base.layers[:-4]:  # Freeze all but last 4 layers
        #     layer.trainable = False
        
        # Classification head for our 4 classes
        # ViT returns 2D features (batch, features), not 3D
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(
                512,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(
                num_classes,
                activation='softmax',
                name='classification_output'
            )
        ])
        
        logging.info(f"Created ProperViTBase for {num_classes} classes")
    
    def call(self, inputs, training=None):
        """Forward pass with image input only."""
        # Ensure inputs are in correct format and range
        if len(inputs.shape) == 4:
            images = inputs
        else:
            raise ValueError(f"Expected 4D image tensor, got shape: {inputs.shape}")
        
        # Ensure images are in [0, 1] range (ViT expects this)
        # Always normalize to be safe in graph mode
        images = tf.cast(images, tf.float32) / 255.0
        
        # ViT forward pass - returns features before classification
        vit_features = self.vit_base(images, training=training)
        
        # Our classification head
        logits = self.classifier(vit_features, training=training)
        
        return logits
    
    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'num_classes': self.num_classes,
            'dropout': self.dropout
        })
        return config

def create_proper_vit_model(
    image_size: int = 224,
    num_classes: int = 4,
    dropout: float = 0.1
) -> ProperViTBase:
    """Create a proper ViT-Base model with correct parameter count."""
    
    model = ProperViTBase(
        image_size=image_size,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Build the model by running a forward pass
    dummy_input = tf.random.normal((1, image_size, image_size, 3))
    _ = model(dummy_input)
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])
    
    logging.info("Proper ViT-Base Model Architecture:")
    logging.info(f"  - Input: ({image_size}, {image_size}, 3) images only")
    logging.info(f"  - Base: ViT-Base-16 with pretrained weights")
    logging.info(f"  - Classification: GAP → Dense(512) → Dense({num_classes})")
    logging.info(f"  - Dropout: {dropout}")
    logging.info(f"  - Total parameters: {total_params:,}")
    logging.info(f"  - Trainable parameters: {trainable_params:,}")
    
    return model

def compile_model(
    model: ProperViTBase,
    learning_rate: float = 3e-4,
    class_weights: Optional[dict] = None
) -> ProperViTBase:
    """Compile model with optimizer and loss."""
    
    # Learning rate schedule - start lower for pretrained model
    learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=1000,
        t_mul=1.0,
        m_mul=1.0,
        alpha=0.1
    )
    
    # AdamW optimizer - good for ViT
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=0.01,
        clipnorm=1.0
    )
    
    # Loss function
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

# Alternative: Use TensorFlow Hub ViT-Base (if vit-keras doesn't work)
def create_hub_vit_model(
    image_size: int = 224,
    num_classes: int = 4,
    dropout: float = 0.1
) -> tf.keras.Model:
    """Alternative: Create ViT using TensorFlow Hub."""
    try:
        import tensorflow_hub as hub
        
        # ViT-Base from TensorFlow Hub
        vit_base_url = "https://tfhub.dev/google/vit_base_patch16_224/1"
        
        inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        
        # Preprocess images for ViT
        x = tf.cast(inputs, tf.float32) / 255.0
        
        # ViT backbone
        vit_base = hub.KerasLayer(vit_base_url, trainable=True)
        features = vit_base(x)
        
        # Classification head
        x = tf.keras.layers.Dropout(dropout)(features)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        logging.info(f"Created TensorFlow Hub ViT-Base with {model.count_params():,} parameters")
        return model
        
    except ImportError:
        logging.error("TensorFlow Hub not available")
        return None

if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)
    
    print("Creating Proper ViT-Base model...")
    model = create_proper_vit_model()
    
    print("\nCompiling model...")
    model = compile_model(model)
    
    print("\nModel ready for training!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Test forward pass
    test_input = tf.random.normal((2, 224, 224, 3))
    output = model(test_input)
    print(f"Test output shape: {output.shape}")
    print("Proper ViT-Base test complete!")