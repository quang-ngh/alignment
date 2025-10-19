#!/bin/bash
# Script to monitor the download progress

# Get the job ID from command line or find the latest
if [ -z "$1" ]; then
    JOB_ID=$(squeue -u $USER -h -o "%i" | head -n1)
else
    JOB_ID=$1
fi

if [ -z "$JOB_ID" ]; then
    echo "No running jobs found."
    exit 1
fi

echo "Monitoring job: $JOB_ID"
echo "Press Ctrl+C to stop monitoring"
echo "========================================"

# Monitor loop
while true; do
    # Clear screen
    clear
    
    echo "Download Monitor - $(date)"
    echo "========================================"
    
    # Check job status
    echo "Job Status:"
    squeue -j $JOB_ID 2>/dev/null || echo "Job completed or not found"
    echo ""
    
    # Check output file
    OUTPUT_FILE="download_${JOB_ID}.out"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Recent Output:"
        echo "----------------------------------------"
        tail -n 20 "$OUTPUT_FILE"
        echo "----------------------------------------"
    fi
    
    # Check disk usage in scratch
    echo ""
    echo "Disk Usage in /scratch/datasets/$USER:"
    if [ -d "/scratch/datasets/$USER" ]; then
        du -sh /scratch/datasets/$USER/* 2>/dev/null | tail -5
        echo ""
        df -h /scratch
    fi
    
    # Check for errors
    ERROR_FILE="download_${JOB_ID}.err"
    if [ -f "$ERROR_FILE" ] && [ -s "$ERROR_FILE" ]; then
        echo ""
        echo "⚠️  Recent Errors:"
        echo "----------------------------------------"
        tail -n 10 "$ERROR_FILE"
        echo "----------------------------------------"
    fi
    
    # Check download.log if it exists
    LOG_FILE="/scratch/datasets/$USER/fifa_pickapic/download.log"
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "Download Progress:"
        echo "----------------------------------------"
        grep -E "GB|percentage|completed|failed" "$LOG_FILE" | tail -5
    fi
    
    # Sleep before next update
    sleep 30
done