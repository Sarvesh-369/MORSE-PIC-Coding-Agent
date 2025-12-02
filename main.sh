#!/bin/bash
# This script is a wrapper for a Python script named 'main.py'.
# It defines default values for various parameters that the Python script might use.
# Example usage:
# ./main.sh --pid 100 --provider qwen --max-iters 5 --stop-on-pass --data custom_data.parquet --images /path/to/images

# Default values for parameters.
PID="37" # A process ID or identifier.
PROVIDER="qwen" # The name of a service provider (e.g., an AI model provider).
MAX_ITERS="2" # Maximum number of iterations for some process.
VERIFICATION_PASSES="1" # Number of verification passes.
DATA="data/testmini.parquet" # Path to a data file, likely in Parquet format.
IMAGES="data/images" # Path to an images directory.

# This section parses command-line arguments passed to the shell script itself.
# It allows users to override the default values defined above.
while [[ "$#" -gt 0 ]]; do # Loop as long as there are arguments.
    case $1 in # Check the current argument.
        --pid) PID="$2"; shift ;; # If it's --pid, assign the next argument ($2) to PID, then shift arguments.
        --provider) PROVIDER="$2"; shift ;; # Same for --provider.
        --max-iters) MAX_ITERS="$2"; shift ;; # Same for --max-iters.
        --verification-passes) VERIFICATION_PASSES="$2"; shift ;; # Same for --verification-passes.
        --stop-on-pass) STOP_ON_PASS="--stop-on-pass" ;; # If --stop-on-pass is present, set a flag variable. Note: it doesn't consume a value.
        --data) DATA="$2"; shift ;; # Same for --data.
        --images) IMAGES="$2"; shift ;; # Same for --images.
        *) echo "Unknown parameter passed: $1"; exit 1 ;; # If an unknown argument is found, print an error and exit.
    esac
    shift # Move to the next argument (or the next pair of argument/value).
done

# Finally, the script executes the 'main.py' Python script.
# It passes all the collected (default or overridden) parameters as command-line arguments to the Python script.
# Each parameter is passed with its corresponding flag (e.g., --pid, --provider).
# The '$STOP_ON_PASS' variable is included as a flag if it was set during argument parsing.
python main.py \
    --pid "$PID" \
    --provider "$PROVIDER" \
    --max-iters "$MAX_ITERS" \
    --verification-passes "$VERIFICATION_PASSES" \
    $STOP_ON_PASS \
    --data "$DATA" \
    --images "$IMAGES"
