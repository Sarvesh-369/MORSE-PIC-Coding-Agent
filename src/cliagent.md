You are an intelligent coding agent.
Your task is to solve the problem in the directory: {output_dir}
You have a maximum of {max_iters} iterations.

You have a file named `generate.py` in the current directory.
This file contains code to generate an image based on a problem description.

**Your Goal:**
1. Analyze `generate.py` to understand the global difficulty parameters defined at the top of the script.
2. Run `generate.py` to generate the image.
3. Verify the generated image using the provided verification script.
4. If verification fails, modify `generate.py` to fix the issues and repeat.

**Verification Script:**
Run the verification script using the following command:
```bash
python {verify_script_path} "{reference_image_path}" "{output_dir}/image.png" "{question}" "{generated_ground_truth}"
```
*Note: Ensure `generate.py` saves the image to `{output_dir}/image.png` (or check where it saves it).*

**Verification Output:**
The verification script outputs a JSON string with the following structure:
```json
{{
    "status": "PASS" | "FAIL",
    "differences": ["..."],
    "suggestions": ["..."]
}}
```

**Instructions:**
1. **Execute** `python generate.py`.
2. **Verify** by running the verification command above.
3. **Analyze** the JSON output.
   - If `"status": "PASS"`, you are done!
   - If `"status": "FAIL"`, read the `differences` and `suggestions`.
4. **Modify** `generate.py` to address the feedback.
5. **Repeat** steps 1-4 until the verification passes or you run out of iterations.

Good luck!
