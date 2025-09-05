#!/bin/bash
# Checkpoint Management Script for GINO Training
# This script helps manage checkpoints for the HPC training jobs

RESULTS_BASE_DIR="/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO"

# Function to list all training runs
list_runs() {
    echo "Available training runs:"
    if [ -d "$RESULTS_BASE_DIR" ]; then
        find "$RESULTS_BASE_DIR" -maxdepth 1 -type d -name "gino_*" | sort
    else
        echo "No training runs found."
    fi
}

# Function to show checkpoints for a specific run
show_checkpoints() {
    local run_dir="$1"
    if [ -z "$run_dir" ]; then
        echo "Usage: show_checkpoints <run_directory>"
        return 1
    fi
    
    echo "Checkpoints in $run_dir:"
    if [ -d "$run_dir/checkpoints" ]; then
        ls -la "$run_dir/checkpoints/"
    else
        echo "No checkpoints directory found."
    fi
}

# Function to find the latest checkpoint
latest_checkpoint() {
    local latest_run=$(find "$RESULTS_BASE_DIR" -maxdepth 1 -type d -name "gino_*" | sort | tail -1)
    if [ -n "$latest_run" ] && [ -f "$latest_run/checkpoints/latest_checkpoint.pth" ]; then
        echo "Latest checkpoint: $latest_run/checkpoints/latest_checkpoint.pth"
        
        # Try to extract epoch info from checkpoint
        if command -v python3 &> /dev/null; then
            echo "Checkpoint details:"
            python3 -c "
import torch
try:
    checkpoint = torch.load('$latest_run/checkpoints/latest_checkpoint.pth', map_location='cpu')
    print(f'  Epoch: {checkpoint[\"epoch\"]} (0-indexed)')
    print(f'  Training losses: {len(checkpoint.get(\"train_losses\", []))} epochs')
    print(f'  Validation losses: {len(checkpoint.get(\"val_losses\", []))} epochs')
    if checkpoint.get('train_losses'):
        print(f'  Latest training loss: {checkpoint[\"train_losses\"][-1]:.6f}')
    if checkpoint.get('val_losses'):
        print(f'  Latest validation loss: {checkpoint[\"val_losses\"][-1]:.6f}')
except Exception as e:
    print(f'  Error reading checkpoint: {e}')
"
        fi
    else
        echo "No latest checkpoint found."
    fi
}

# Function to clean old checkpoints (keep latest N)
clean_checkpoints() {
    local run_dir="$1"
    local keep="${2:-3}"  # Default: keep 3 checkpoints
    
    if [ -z "$run_dir" ]; then
        echo "Usage: clean_checkpoints <run_directory> [number_to_keep]"
        return 1
    fi
    
    if [ ! -d "$run_dir/checkpoints" ]; then
        echo "No checkpoints directory found in $run_dir"
        return 1
    fi
    
    echo "Cleaning old checkpoints in $run_dir/checkpoints (keeping $keep newest)..."
    
    # Find checkpoint files (excluding latest and final)
    cd "$run_dir/checkpoints"
    checkpoint_files=$(ls -1 checkpoint_epoch_*.pth 2>/dev/null | sort -V)
    total_files=$(echo "$checkpoint_files" | wc -l)
    
    if [ "$total_files" -gt "$keep" ]; then
        to_delete=$(echo "$checkpoint_files" | head -n $((total_files - keep)))
        echo "Deleting old checkpoints:"
        echo "$to_delete"
        echo "$to_delete" | xargs rm -f
        echo "Cleanup completed."
    else
        echo "No cleanup needed. Found $total_files checkpoint files."
    fi
}

# Function to backup important checkpoints
backup_checkpoint() {
    local checkpoint_path="$1"
    local backup_dir="${2:-$HOME/checkpoint_backups}"
    
    if [ -z "$checkpoint_path" ]; then
        echo "Usage: backup_checkpoint <checkpoint_path> [backup_directory]"
        return 1
    fi
    
    if [ ! -f "$checkpoint_path" ]; then
        echo "Checkpoint file not found: $checkpoint_path"
        return 1
    fi
    
    mkdir -p "$backup_dir"
    local backup_name="$(basename $(dirname $(dirname $checkpoint_path)))_$(basename $checkpoint_path)"
    cp "$checkpoint_path" "$backup_dir/$backup_name"
    echo "Checkpoint backed up to: $backup_dir/$backup_name"
}

# Function to show disk usage
disk_usage() {
    if [ -d "$RESULTS_BASE_DIR" ]; then
        echo "Disk usage for training results:"
        du -h --max-depth=2 "$RESULTS_BASE_DIR" | sort -hr
    else
        echo "Results directory not found."
    fi
}

# Main script logic
case "$1" in
    "list"|"ls")
        list_runs
        ;;
    "show")
        show_checkpoints "$2"
        ;;
    "latest")
        latest_checkpoint
        ;;
    "clean")
        clean_checkpoints "$2" "$3"
        ;;
    "backup")
        backup_checkpoint "$2" "$3"
        ;;
    "usage"|"du")
        disk_usage
        ;;
    *)
        echo "Checkpoint Management Script for GINO Training"
        echo ""
        echo "Usage: $0 <command> [arguments]"
        echo ""
        echo "Commands:"
        echo "  list, ls                     - List all training runs"
        echo "  show <run_directory>         - Show checkpoints for a specific run"
        echo "  latest                       - Show latest checkpoint with details"
        echo "  clean <run_dir> [keep_count] - Clean old checkpoints (default: keep 3)"
        echo "  backup <checkpoint> [dir]    - Backup a checkpoint to safe location"
        echo "  usage, du                    - Show disk usage of results"
        echo ""
        echo "Examples:"
        echo "  $0 list"
        echo "  $0 latest"
        echo "  $0 show /srv/scratch/.../gino_20240101_120000"
        echo "  $0 clean /srv/scratch/.../gino_20240101_120000 5"
        echo "  $0 backup /path/to/checkpoint.pth"
        ;;
esac
