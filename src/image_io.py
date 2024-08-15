# src/image_io.py

import os
import numpy as np
import tifffile as tif
from tkinter import Tk, filedialog
from typing import Tuple, Optional


def load_segmented_image() -> Tuple[Optional[str], Optional[str], Optional[np.ndarray]]:
    """
    Load a segmented numpy image file (.npy output from Cellpose 2.0) selected via a file dialog.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[np.ndarray]]: A tuple containing the directory path,
        filename, and segmented image array. Returns (None, None, None) if no file is selected.
    """
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()  # Open a file dialog to select the file
    if not file_path:
        print("No file selected.")
        return None, None, None
    dir_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_path}")
    segmented_data = np.load(file_path, allow_pickle=True).item()  # Load the numpy file
    segmented_image = segmented_data['masks']  # Extract the 'masks' array
    return dir_path, file_name, segmented_image

def save_image(directory: str, file_name: str, image: np.ndarray, suffix: str) -> None:
    """
    Save the image to the file system with a specific suffix.

    Args:
        directory (str): Directory path where the image will be saved.
        file_name (str): Original filename of the image.
        image (np.ndarray): Image array to save.
        suffix (str): Suffix to append to the filename.
    """
    full_path = os.path.join(directory, f"{os.path.splitext(file_name)[0]}_{suffix}.tif")
    tif.imwrite(full_path, image)
    print(f"Image saved to {full_path}")

def create_pif_file(relabeled_image: np.ndarray, dir_path: str, file_name: str) -> None:
    """
    Create a PIF file from the relabeled image for simulation purposes.

    Args:
        relabeled_image (np.ndarray): The relabeled image.
        dir_path (str): Directory path for file operations.
        file_name (str): Original filename of the image.
    """
    import timeit
    
    start_time = timeit.default_timer()

    pif_data = []
    # Iterate over each pixel in the label image
    for z in range(relabeled_image.shape[0]):
        for y in range(relabeled_image.shape[1]):
            for x in range(relabeled_image.shape[2]):
                pixel_value = relabeled_image[z, y, x]

                # Check if the pixel value is not 0
                if pixel_value != 0:
                    pif_data.append((pixel_value, 'A', x, x, y, y, z, z))

    end_time = timeit.default_timer()
    print("Time to create PIF data: ", end_time - start_time, "seconds")

    start_time = timeit.default_timer()
    # Write the PIF data to a file
    pif_file_path = os.path.join(dir_path, "cell_layout_simulation.piff")
    with open(pif_file_path, 'w') as output_file:
        for item in pif_data:
            line = ' '.join(str(value) for value in item)
            output_file.write(line + '\n')

    end_time = timeit.default_timer()
    print("Total time to write PIF file: ", end_time - start_time, "seconds")
