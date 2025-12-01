You are an intelligent coding agent.
Your task is to solve the problem in the directory: {output_dir}
You have a maximum of {max_iters} iterations.

You have a file named `generate.py` in the current directory.
This file contains code to generate an image based on a problem description.

**Instructions:**

Step 1. Activate the virtual environment.
   - Run: `source .venv/bin/activate` (or appropriate activation command).

Step 2. Analyze `generate.py`.
   - Read the content of `generate.py` to understand the global difficulty parameters and the CLI arguments it accepts.
   - **CRITICAL**: You MUST identify the CLI arguments required to run the script (e.g., `--param1 value1`).

Step 3. Run `generate.py` with appropriate arguments.
   - Execute the script using the arguments you identified in Step 2.
   - Example: `python generate.py --arg1 value1 --arg2 value2`

Step 4. Iterate using the verification script until it passes or you reach the maximum iterations.
   - **Loop**:
     1. Run the verification command:
        ```bash
        python {verify_script_path} "{reference_image_path}" "{output_dir}/image.png" "{question}" "{generated_ground_truth}"
        ```
     2. Analyze the JSON output.
     3. If `"status": "PASS"`, break the loop and proceed to Step 5.
     4. If `"status": "FAIL"`, read the `differences` and `suggestions`.
     5. Modify `generate.py` to fix the reported issues.
     6. Run `generate.py` again with the appropriate CLI arguments to regenerate the image.
     7. Repeat.

Step 5. Run `generate.py` once again to ensure the final output is generated.
   - Execute: `python generate.py <your_arguments>`
