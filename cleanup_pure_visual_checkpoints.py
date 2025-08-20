#!/usr/bin/env python3
"""
Pure Visual Checkpoint Cleanup Monitor for NQ_AI Training
Maintains minimal checkpoints during pure visual training to prevent disk space issues
"""

import os
import time
import glob
import re
import logging
from pathlib import Path

def setup_logging():
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/pure_visual_checkpoint_cleanup.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_file_size_gb(filepath):
    """Get file size in GB."""
    try:
        return os.path.getsize(filepath) / (1024**3)
    except:
        return 0.0

def cleanup_pure_visual_checkpoints(checkpoint_dir, logger):
    """Clean up pure visual checkpoints to save disk space."""
    try:
        # Check for various pure visual checkpoint patterns
        patterns = [
            os.path.join(checkpoint_dir, 'pure_visual_daily_model_*.weights.h5'),
            os.path.join(checkpoint_dir, 'pure_visual_daily_*.weights.h5'),
            os.path.join(checkpoint_dir, '*pure_visual*.weights.h5')
        ]
        
        all_checkpoints = []
        for pattern in patterns:
            all_checkpoints.extend(glob.glob(pattern))
        
        # Remove duplicates and sort by modification time (newest first)
        unique_checkpoints = list(set(all_checkpoints))
        if not unique_checkpoints:
            logger.debug("No pure visual checkpoint files found")
            return
        
        # Sort by modification time (newest first)
        checkpoints_with_time = []
        for filepath in unique_checkpoints:
            try:
                mtime = os.path.getmtime(filepath)
                size_gb = get_file_size_gb(filepath)
                checkpoints_with_time.append((filepath, mtime, size_gb))
            except Exception as e:
                logger.warning(f"Error accessing {filepath}: {e}")
        
        checkpoints_with_time.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the newest 2 checkpoints, delete the rest
        keep_count = 2
        if len(checkpoints_with_time) <= keep_count:
            total_size = sum([x[2] for x in checkpoints_with_time])
            logger.info(f"Found {len(checkpoints_with_time)} checkpoints ({total_size:.1f}GB total) - no cleanup needed")
            return
        
        # Keep the newest checkpoints
        keep_checkpoints = checkpoints_with_time[:keep_count]
        delete_checkpoints = checkpoints_with_time[keep_count:]
        
        logger.info(f"Keeping {len(keep_checkpoints)} newest checkpoints:")
        for filepath, mtime, size_gb in keep_checkpoints:
            filename = os.path.basename(filepath)
            logger.info(f"  KEEP: {filename} ({size_gb:.1f}GB)")
        
        # Delete old checkpoints
        total_freed = 0
        for filepath, mtime, size_gb in delete_checkpoints:
            try:
                filename = os.path.basename(filepath)
                os.remove(filepath)
                total_freed += size_gb
                logger.info(f"  DELETED: {filename} - freed {size_gb:.1f}GB")
            except Exception as e:
                logger.error(f"Error deleting {filepath}: {e}")
        
        if total_freed > 0:
            logger.info(f"Total space freed: {total_freed:.1f}GB")
            
    except Exception as e:
        logger.error(f"Error during pure visual checkpoint cleanup: {e}")

def monitor_disk_usage(checkpoint_dir, logger):
    """Monitor disk usage and warn if space is low."""
    try:
        statvfs = os.statvfs(checkpoint_dir)
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        total_space_gb = (statvfs.f_frsize * statvfs.f_blocks) / (1024**3)
        used_percent = ((total_space_gb - free_space_gb) / total_space_gb) * 100
        
        logger.info(f"Disk usage: {used_percent:.1f}% used, {free_space_gb:.1f}GB free")
        
        if free_space_gb < 5.0:  # Less than 5GB free
            logger.warning(f"LOW DISK SPACE: Only {free_space_gb:.1f}GB remaining!")
        elif free_space_gb < 10.0:  # Less than 10GB free
            logger.warning(f"Disk space getting low: {free_space_gb:.1f}GB remaining")
            
    except Exception as e:
        logger.error(f"Error checking disk usage: {e}")

def monitor_pure_visual_checkpoints(checkpoint_dir, check_interval=600):  # 10 minutes
    """Monitor and cleanup pure visual checkpoints periodically."""
    logger = setup_logging()
    logger.info("Starting pure visual checkpoint cleanup monitor")
    logger.info(f"Monitoring directory: {checkpoint_dir}")
    logger.info(f"Check interval: {check_interval} seconds ({check_interval/60:.1f} minutes)")
    logger.info("Strategy: Keep 2 newest checkpoints, delete older ones")
    
    while True:
        try:
            if os.path.exists(checkpoint_dir):
                # Monitor disk usage
                monitor_disk_usage(checkpoint_dir, logger)
                
                # Clean up old checkpoints
                cleanup_pure_visual_checkpoints(checkpoint_dir, logger)
            else:
                logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
                logger.info("Waiting for training to create checkpoint directory...")
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            logger.info("Pure visual checkpoint monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in monitor: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    # Pure visual checkpoint directory
    checkpoint_dir = "models/outputs_pure_visual/checkpoints"
    monitor_pure_visual_checkpoints(checkpoint_dir)