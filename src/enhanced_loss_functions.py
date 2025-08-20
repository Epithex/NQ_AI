#!/usr/bin/env python3
"""
Enhanced Loss Functions for Improving High Breakout and Low Breakdown Performance
Specialized loss functions that focus on the two most common patterns
"""

import tensorflow as tf
import numpy as np


class PatternSpecificFocalLoss(tf.keras.losses.Loss):
    """
    Focal loss with pattern-specific weighting for High Breakout and Low Breakdown
    """
    
    def __init__(self, 
                 alpha_high_breakout=0.4,
                 alpha_low_breakdown=0.35, 
                 alpha_range_expansion=0.15,
                 alpha_range_bound=0.1,
                 gamma_main=3.0,
                 gamma_rare=2.0,
                 name="pattern_specific_focal_loss"):
        super().__init__(name=name)
        self.alpha_high_breakout = alpha_high_breakout
        self.alpha_low_breakdown = alpha_low_breakdown
        self.alpha_range_expansion = alpha_range_expansion
        self.alpha_range_bound = alpha_range_bound
        self.gamma_main = gamma_main
        self.gamma_rare = gamma_rare
        
        # Create alpha tensor
        self.alphas = tf.constant([
            alpha_high_breakout,   # Class 0: High Breakout
            alpha_low_breakdown,   # Class 1: Low Breakdown
            alpha_range_expansion, # Class 2: Range Expansion
            alpha_range_bound      # Class 3: Range Bound
        ], dtype=tf.float32)
        
        # Create gamma tensor (higher for main patterns)
        self.gammas = tf.constant([
            gamma_main,  # Class 0: High Breakout
            gamma_main,  # Class 1: Low Breakdown  
            gamma_rare,  # Class 2: Range Expansion
            gamma_rare   # Class 3: Range Bound
        ], dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        """
        Compute pattern-specific focal loss
        """
        # Convert to int32 for indexing
        y_true_int = tf.cast(y_true, tf.int32)
        
        # Get alpha values for each sample
        alpha_t = tf.gather(self.alphas, y_true_int)
        
        # Get gamma values for each sample
        gamma_t = tf.gather(self.gammas, y_true_int)
        
        # Compute cross entropy
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        
        # Get predicted probability for true class
        p_t = tf.gather(y_pred, y_true_int, batch_dims=1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1 - p_t, gamma_t)
        
        # Apply alpha weighting and focal weight
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return tf.reduce_mean(focal_loss)


class ConfidenceAwareLoss(tf.keras.losses.Loss):
    """
    Loss function that penalizes confident wrong predictions more heavily on main patterns
    """
    
    def __init__(self, main_pattern_penalty=2.0, name="confidence_aware_loss"):
        super().__init__(name=name)
        self.main_pattern_penalty = main_pattern_penalty
    
    def call(self, y_true, y_pred):
        """
        Compute confidence-aware loss with higher penalties for main patterns
        """
        # Base cross-entropy loss
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        
        # Get prediction confidence (max probability)
        confidence = tf.reduce_max(y_pred, axis=-1)
        
        # Get predicted class
        y_pred_class = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true_int = tf.cast(y_true, tf.int32)
        
        # Check if prediction is wrong
        wrong_prediction = tf.not_equal(y_pred_class, y_true_int)
        
        # Check if true class is main pattern (0 or 1)
        main_pattern_mask = tf.logical_or(
            tf.equal(y_true_int, 0),  # High Breakout
            tf.equal(y_true_int, 1)   # Low Breakdown
        )
        
        # Apply higher penalty for confident wrong predictions on main patterns
        confidence_penalty = tf.where(
            tf.logical_and(wrong_prediction, main_pattern_mask),
            1.0 + confidence * self.main_pattern_penalty,  # Higher penalty for main patterns
            1.0 + confidence * 0.5                        # Lower penalty for rare patterns
        )
        
        return tf.reduce_mean(base_loss * confidence_penalty)


class MainPatternWeightedLoss(tf.keras.losses.Loss):
    """
    Weighted loss that gives extra importance to High Breakout and Low Breakdown accuracy
    """
    
    def __init__(self, 
                 main_pattern_weight=1.5,
                 rare_pattern_weight=1.0,
                 name="main_pattern_weighted_loss"):
        super().__init__(name=name)
        self.main_pattern_weight = main_pattern_weight
        self.rare_pattern_weight = rare_pattern_weight
        
        # Create weight tensor
        self.class_weights = tf.constant([
            main_pattern_weight,  # Class 0: High Breakout
            main_pattern_weight,  # Class 1: Low Breakdown
            rare_pattern_weight,  # Class 2: Range Expansion
            rare_pattern_weight   # Class 3: Range Bound
        ], dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        """
        Compute weighted cross-entropy with emphasis on main patterns
        """
        # Base cross-entropy loss
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        
        # Get weights for each sample
        y_true_int = tf.cast(y_true, tf.int32)
        sample_weights = tf.gather(self.class_weights, y_true_int)
        
        # Apply weights
        weighted_loss = base_loss * sample_weights
        
        return tf.reduce_mean(weighted_loss)


def create_combined_loss(focal_weight=0.4, confidence_weight=0.3, weighted_weight=0.3):
    """
    Create a combined loss function that uses all three enhanced losses
    """
    focal_loss = PatternSpecificFocalLoss()
    confidence_loss = ConfidenceAwareLoss()
    weighted_loss = MainPatternWeightedLoss()
    
    def combined_loss(y_true, y_pred):
        focal_component = focal_loss(y_true, y_pred)
        confidence_component = confidence_loss(y_true, y_pred)
        weighted_component = weighted_loss(y_true, y_pred)
        
        return (focal_weight * focal_component + 
                confidence_weight * confidence_component + 
                weighted_weight * weighted_component)
    
    return combined_loss


# Convenience functions for easy import
def get_pattern_specific_focal_loss():
    """Get pattern-specific focal loss optimized for main patterns"""
    return PatternSpecificFocalLoss(
        alpha_high_breakout=0.4,
        alpha_low_breakdown=0.35,
        alpha_range_expansion=0.15,
        alpha_range_bound=0.1,
        gamma_main=3.0,
        gamma_rare=2.0
    )


def get_confidence_aware_loss():
    """Get confidence-aware loss with main pattern focus"""
    return ConfidenceAwareLoss(main_pattern_penalty=2.0)


def get_main_pattern_weighted_loss():
    """Get weighted loss emphasizing main patterns"""
    return MainPatternWeightedLoss(
        main_pattern_weight=1.5,
        rare_pattern_weight=1.0
    )


def get_combined_loss():
    """Get combined loss function with balanced components"""
    return create_combined_loss(
        focal_weight=0.4,
        confidence_weight=0.3, 
        weighted_weight=0.3
    )