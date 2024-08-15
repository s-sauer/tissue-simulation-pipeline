# tissue-simulation-pipeline

This repo contains code + additional material to run the simulations published in [E Nürnberg et al.: From in vitro to in silico: a pipeline for generating virtual tissue simulations from real image data (2024), Preprint on bioRxiv.org]([www.google.de](https://www.biorxiv.org/content/10.1101/2024.07.12.603259v1)

## Content:
- `/src` Python code to run the pipeline
- `/src/main.py` the main routine. Execute this Python script to start the pipeline
- `Biophysical_simulation.zip` contains all files required to execute the CompuCell3D simulation
- `intensity_image.tif` 3D confocal image of the spheroid shown in the publication
- `segmentation_seg.npy` numpy array with segmentation results for this spheroid
