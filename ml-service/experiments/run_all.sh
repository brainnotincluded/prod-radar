#!/bin/bash
# Run all 10 experiment variants sequentially
# Usage: cd ~/prod-radar-ml && bash experiments/run_all.sh

export LD_LIBRARY_PATH=/home/ubuntu/prod-radar-ml/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
source venv/bin/activate

RESULTS_FILE="experiments/results_summary.txt"
echo "EXPERIMENT RESULTS SUMMARY" > $RESULTS_FILE
echo "Started: $(date)" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE

# Fast experiments first (tiny models, ~8-15 min each)
for script in \
    experiments/v1_tiny_domain.py \
    experiments/v9_preprocessing.py \
    experiments/v7_curriculum.py \
    experiments/v5_clean_data.py \
    experiments/v10_multitask.py \
    experiments/v2_hierarchical.py \
    experiments/v4_ensemble_tiny.py \
    experiments/v8_rubert_base.py \
    experiments/v6_distillation.py \
    experiments/v3_softmax_smooth.py; do

    name=$(basename $script .py)
    echo ""
    echo "========================================"
    echo "RUNNING: $name"
    echo "========================================"
    echo ""

    start_time=$(date +%s)
    python3 $script 2>&1 | tee experiments/${name}.log
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Extract F1-macro from log
    f1=$(grep -oP 'F1-macro[:\s]+\K[0-9.]+' experiments/${name}.log | tail -1)

    echo "" >> $RESULTS_FILE
    echo "$name: F1-macro=$f1 (${duration}s, exit=$exit_code)" >> $RESULTS_FILE
done

echo "" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "Finished: $(date)" >> $RESULTS_FILE

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"
cat $RESULTS_FILE
