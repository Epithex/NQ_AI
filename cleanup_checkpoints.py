#!/usr/bin/env python3
"""
Checkpoint Cleanup Monitor for NQ_AI Training
Maintains only the best checkpoint to prevent disk space issues
"""

import os
import time
import glob
import re
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/checkpoint_cleanup.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_accuracy_from_filename(filename):
    """Extract validation accuracy from checkpoint filename."""
    match = re.search(r'acc_(\d+\.\d+)\.weights\.h5$', filename)
    return float(match.group(1)) if match else 0.0

def cleanup_old_checkpoints(checkpoint_dir, logger):
    """Keep only the checkpoint with highest validation accuracy."""
    try:
        # Find all checkpoint files
        pattern = os.path.join(checkpoint_dir, 'hybrid_daily_vit_epoch_*_acc_*.weights.h5')
        checkpoint_files = glob.glob(pattern)
        
        if len(checkpoint_files) <= 1:
            logger.info(f"Found {len(checkpoint_files)} checkpoints - no cleanup needed")
            return
        
        # Extract accuracy for each file
        checkpoints_with_acc = []
        for filepath in checkpoint_files:
            filename = os.path.basename(filepath)
            accuracy = extract_accuracy_from_filename(filename)
            checkpoints_with_acc.append((filepath, accuracy, filename))
        
        # Sort by accuracy (highest first)
        checkpoints_with_acc.sort(key=lambda x: x[1], reverse=True)
        
        # Keep the best one, delete the rest
        best_checkpoint = checkpoints_with_acc[0]
        old_checkpoints = checkpoints_with_acc[1:]
        
        logger.info(f"Best checkpoint: {best_checkpoint[2]} (acc: {best_checkpoint[1]:.4f})")
        
        # Delete old checkpoints
        total_freed = 0
        for filepath, accuracy, filename in old_checkpoints:
            try:
                file_size = os.path.getsize(filepath) / (1024**3)  # GB
                os.remove(filepath)
                total_freed += file_size
                logger.info(f"Deleted: {filename} (acc: {accuracy:.4f}) - freed {file_size:.1f}GB")
            except Exception as e:
                logger.error(f"Error deleting {filename}: {e}")
        
        if total_freed > 0:
            logger.info(f"Total space freed: {total_freed:.1f}GB")
            
    except Exception as e:
        logger.error(f"Error during checkpoint cleanup: {e}")

def monitor_checkpoints(checkpoint_dir, check_interval=300):  # 5 minutes
    """Monitor and cleanup checkpoints periodically."""
    logger = setup_logging()
    logger.info("Starting checkpoint cleanup monitor")
    logger.info(f"Monitoring directory: {checkpoint_dir}")
    logger.info(f"Check interval: {check_interval} seconds")
    
    while True:
        try:
            if os.path.exists(checkpoint_dir):
                cleanup_old_checkpoints(checkpoint_dir, logger)
            else:
                logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            logger.info("Checkpoint monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in monitor: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    # Default checkpoint directory
    checkpoint_dir = "models/outputs_daily_v2/checkpoints"
    monitor_checkpoints(checkpoint_dir)