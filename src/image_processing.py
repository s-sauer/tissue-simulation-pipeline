# src/image_processing.py

import numpy as np
from skimage.morphology import label
from scipy.ndimage import zoom
from typing import List, Tuple

def find_labels_in_few_planes(segmented_image: np.ndarray) -> List[int]:
    """
    Identify labels that appear in two or fewer z-planes.

    Args:
        segmented_image (np.ndarray): 3D array representing the segmented image.

    Returns:
        List[int]: List of labels that appear in two or fewer z-planes.
    """
    import pandas as pd
    from skimage.measure import regionprops_table

    properties = ['label', 'image']
    # Extract properties of each labeled region
    single_cell_masks = pd.DataFrame(regionprops_table(segmented_image, intensity_image=segmented_image, properties=properties))
    single_cell_masks = single_cell_masks.set_index('label')
    # Calculate the number of z-planes for each label
    for index, row in single_cell_masks.iterrows():
        label_mask = row['image']
        single_cell_masks.at[index, 'z_planes_count'] = label_mask.shape[0]
    single_cell_masks['z_planes_count'] = single_cell_masks['z_planes_count'].astype(int)
    labels_to_change = single_cell_masks[single_cell_masks['z_planes_count'] <= 2].index.tolist()
    print(f"Total number of labels: {len(np.unique(segmented_image))}")
    print(f"Number of labels in 2 or fewer z-planes: {len(labels_to_change)}")
    return labels_to_change

def assign_dummy_label(segmented_image: np.ndarray, labels_to_change: List[int]) -> np.ndarray:
    """
    Change specified labels that appear in two or fewer z-planes in the segmented image to a dummy value (101010).

    Args:
        segmented_image (np.ndarray): 3D array representing the segmented image.
        labels_to_change (List[int]): List of labels to change.

    Returns:
        np.ndarray: The segmented image with specified labels changed to a dummy value.
    """
    modified_image = np.copy(segmented_image).astype(np.int32)
    for label in labels_to_change:
        modified_image[segmented_image == label] = 101010

    # Count the number of voxels with label 101010 before the change
    num_occurrences_before = np.sum(segmented_image == 101010)
    print(f"Number of voxels with dummy label 101010 before the change: {num_occurrences_before}")

    # Count the number of voxels with label 101010 after the change
    num_occurrences_after = np.sum(modified_image == 101010)
    print(f"Number of voxels with dummy label 101010 after the change: {num_occurrences_after}")

    return modified_image

def remove_specified_labels(image: np.ndarray, labels_to_remove: List[int]) -> np.ndarray:
    """
    Remove specified labels from the image by setting their pixels to the background value (0).

    Args:
        image (np.ndarray): 3D array representing the image.
        labels_to_remove (List[int]): List of labels to delete.

    Returns:
        np.ndarray: The image with specified labels removed.
    """
    for label in labels_to_remove:
        image[image == label] = 0
    return image

def relabel_image(image: np.ndarray) -> np.ndarray:
    """
    Relabel the image by assigning new consecutive labels starting from 1, skipping the background (0).

    Args:
        image (np.ndarray): 3D array representing the image to relabel.

    Returns:
        np.ndarray: The relabeled image.
    """
    # Count the number of unique labels before relabeling
    num_unique_labels_before = len(np.unique(image))
    print(f"Number of unique labels before relabeling: {num_unique_labels_before}")

    relabeled_image = label(image, background=0)

    # Count the number of unique labels after relabeling
    num_unique_labels_after = len(np.unique(relabeled_image))
    print(f"Number of unique labels after relabeling: {num_unique_labels_after}")
    return relabeled_image.astype(np.int32)

def resample_image_to_isotropic_voxels(image: np.ndarray, voxel_size: Tuple[float, float, float]) -> np.ndarray:
    """
    Resample the image based on the specified voxel size to achieve isotropic voxels.

    Args:
        image (np.ndarray): 3D array representing the image to resample.
        voxel_size (Tuple[float, float, float]): Tuple specifying the voxel size in the format (z, y, x).

    Returns:
        np.ndarray: The resampled image.
    """
    zoom_factors = (voxel_size[0] / voxel_size[1], 1, 1)
    resampled_image = zoom(image, zoom=zoom_factors, order=0)
    return resampled_image
