# tissue-simulation-pipeline

This repo contains code + additional material to run the simulations published in [E Nürnberg et al.: From in vitro to in silico: a pipeline for generating virtual tissue simulations from real image data, Front. Mol. Biosci. (2024)](https://doi.org/10.3389/fmolb.2024.1467366) and [R Bruch et al.: Improving 3D deep learning segmentation with biophysically motivated cell synthesis, Commun. Biol. (2025)](https://doi.org/10.1038/s42003-025-07469-2)

## Content:
- `/src` Python code to run the pipeline
- `/src/main.py` the main routine. Execute this Python script to start the pipeline
- `Biophysical_simulation.zip` contains all files required to execute the CompuCell3D simulation
- `intensity_image.tif` 3D confocal image of the spheroid shown in the publication
- `segmentation_seg.npy.zip` (zipped) numpy array with segmentation results for this spheroid
