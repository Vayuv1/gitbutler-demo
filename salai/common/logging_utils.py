# salai/asr_finetune/common/logging_utils.py
"""
Utilities for logging, monitoring, and metadata collection during training runs.
"""
import subprocess
import torch
import logging

logger = logging.getLogger(__name__)

def get_git_commit_hash() -> str:
    """Gets the current git commit hash."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).strip().decode("utf-8")
        return commit_hash
    except subprocess.CalledProcessError:
        logger.warning("Could not get git commit hash. Is this a git repository?")
        return "N/A"

def log_gpu_memory(message: str = "GPU Memory Usage"):
    """Logs the current GPU memory usage if a GPU is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"{message}: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")

def get_peak_memory_mb() -> float:
    """
    Returns the peak GPU memory allocated in megabytes since the beginning
    of the run.
    """
    if torch.cuda.is_available():
        # Resets the peak stats, so call this only once at the end of the run.
        peak_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return peak_bytes / 1024**2
    return 0.0
