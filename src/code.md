You are an expert Python programmer and data visualization specialist.
Your task is to generate a Python script `generate.py` that recreates a given image programmatically.

## Step 1 — Analyze the Request
- **Image**: You will be provided with an input image.
- **Context**: You will be provided with the context of the image (e.g., "scientific figure", "bar chart").
- **Description**: Analyze the image and provide a detailed description.

## Step 2 — Choose the Appropriate Library
1. Understand the image and give an image description for it.
2. Understand which context the image belongs to then, Strictly select the **Appropriate Library** for the context from the listed, check is that library exist in the virtual environment and use it:

| Visualization Type | Libraries |
|-------------------|-----------|
| ABSTRACT_SCENE | matplotlib, turtle, PIL, numpy |
| BAR_CHART | matplotlib, seaborn, plotly, pandas |
| DOCUMENT_IMAGE | PIL, matplotlib, sympy, cv2 |
| FUNCTION_PLOT | matplotlib, numpy, sympy, plotly |
| GEOMETRY_DIAGRAM | matplotlib, turtle, PIL, numpy |
| LINE_PLOT | matplotlib, seaborn, plotly, pandas |
| MAP_CHART | folium, plotly, matplotlib, pandas |
| PIE_CHART | matplotlib, plotly, seaborn |
| PUZZLE_TEST | PIL, matplotlib, turtle, numpy |
| SCATTER_PLOT | matplotlib, seaborn, plotly, pandas |
| SCIENTIFIC_FIGURE | matplotlib, plotly, numpy, seaborn |
| SYNTHETIC_SCENE | pyvista, trimesh, open3d |
| TABLE | pandas, matplotlib, PIL, plotly |
| VIOLIN_PLOT | seaborn, matplotlib, plotly, pandas |

**Rules:**
* Use the library thats strictly instructed.
* Prefer built-ins + standard library where possible.
* Do not use any online, external, or cloud services.

## Step 3 — Write `generate.py`
Based on your understanding of the image and the description you came up with:
Programatically create the image using python.

### Requirements

* Use `argparse` to create CLI arguments for all the global difficulty parameters:
  ```bash
  python generate.py --param1 value1 --param2 value2
  ```
* The code should:
  1. **Define global difficulty parameters** at the top of the script that control the complexity of the image, **changing these parameters should also be reflected in the image**: For example:
     - For charts: number of data points, categories, series, etc.
     - For geometry: number of shapes, complexity of figures, etc.
     - For plots: number of functions, data density, etc.
     - For tables: number of rows/columns, etc. 
     (you need not stick with these examples, you can define your own parameters)
  2. Set the random seed using a seed variable (default 42) for reproducibility.
  3. Generate an **image** that recreates the input image based on the context and description. The image MUST be saved as `image.png` in the current working directory.
  4. Compute the **answer** directly from the scene data in the generated program itself.
