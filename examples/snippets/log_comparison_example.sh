#!/bin/bash
# Example: Comparing multiple training runs using logs commands

echo "============================================"
echo "Log Analysis Examples"
echo "============================================"
echo

# Navigate to a project with training logs
cd examples/tiny_experiments/ddp_trainer || exit 1

echo "1. List all training logs:"
echo "-------------------------------------------"
forgather logs list
echo

echo "2. Quick summary of latest run (one-line):"
echo "-------------------------------------------"
forgather logs summary --format one-line
echo

echo "3. Compare all runs in a table:"
echo "-------------------------------------------"
forgather logs summary --all --format one-line
echo

echo "4. Detailed summary of latest run:"
echo "-------------------------------------------"
forgather logs summary | head -30
echo

echo "5. Get all summaries as JSON array:"
echo "-------------------------------------------"
forgather logs summary --all --format json --output /tmp/all_logs_summary.json
echo "Saved to: /tmp/all_logs_summary.json"
echo "First 15 lines:"
head -15 /tmp/all_logs_summary.json
echo

echo "6. Export individual log summaries:"
echo "-------------------------------------------"
for log in output_models/default_model/runs/*/trainer_logs.json; do
    run_name=$(basename $(dirname "$log"))
    echo "Processing: $run_name"
    forgather logs summary "$log" --format one-line
done
echo

echo "============================================"
echo "Examples complete!"
echo "============================================"
