# src/analysis.py

import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import find_boundaries
from scipy.spatial import distance
from scipy.ndimage import zoom
import os
import math
from typing import List, Tuple
import tifffile as tif

def calculate_statistics(image: np.ndarray, properties: List[str], file_path: str, description: str) -> pd.DataFrame:
    """
    Calculate statistics for given properties of labeled regions in an image and save them to a file.

    Args:
        image (np.ndarray): 3D array representing the labeled image.
        properties (List[str]): List of properties to calculate for each region.
        file_path (str): Path to save the statistics file.
        description (str): Description of the statistics for logging.

    Returns:
        pd.DataFrame: DataFrame containing the calculated properties.
    """
    try:
        cell_features = pd.DataFrame(regionprops_table(image, intensity_image=image, properties=properties))
        if cell_features.empty:
            print("No data available for statistics.")
            return pd.DataFrame()

        # Rename columns and modify DataFrame structure
        print(cell_features.columns)
        cell_features.rename(columns={'area': 'volume', 'equivalent_diameter_area': 'equivalent_volume'}, inplace=True)
        cell_features['centroid (z, y, x)'] = list(zip(cell_features['centroid-0'], cell_features['centroid-1'], cell_features['centroid-2']))
        cell_features.drop(['centroid-0', 'centroid-1', 'centroid-2'], axis=1, inplace=True)

        statistics = cell_features.describe()
        statistics.to_csv(file_path, sep='\t', index=True)
        print(f"{description} statistics saved to {file_path}")
    except Exception as e:
        print(f"Error during statistics calculation: {e}")
        return pd.DataFrame()

    return cell_features

def apply_small_volume_threshold(image: np.ndarray, cell_properties_df: pd.DataFrame, threshold_factor: float = 0.15) -> Tuple[np.ndarray, List[int]]:
    """
    Apply a volume threshold to remove cells below a specified percentage of the average volume.

    Args:
        image (np.ndarray): 3D array representing the image.
        cell_properties_df (pd.DataFrame): DataFrame containing cell properties.
        threshold_factor (float): Factor to determine the volume threshold, default is 0.15.

    Returns:
        Tuple[np.ndarray, List[int]]: The modified image and a list of labels removed.
    """
    if cell_properties_df.empty:
        print("No valid cell properties data available for applying volume threshold.")
        return image, []

    mean_volume = cell_properties_df['volume'].mean()
    lower_volume_cutoff = threshold_factor * mean_volume
    print(f"Lower volume cutoff: {lower_volume_cutoff}")

    labels_to_remove = cell_properties_df[cell_properties_df['volume'] < lower_volume_cutoff]['label'].tolist()
    for label in labels_to_remove:
        image[image == label] = 0
    print(f"Applied volume threshold: {lower_volume_cutoff}, removing {len(labels_to_remove)} labels")
    return image, labels_to_remove

def identify_large_volume_labels(image: np.ndarray, cell_properties_df: pd.DataFrame) -> List[int]:
    """
    Identify cells with a volume significantly above the average using an upper volume threshold.

    Args:
        image (np.ndarray): 3D array representing the image.
        cell_properties_df (pd.DataFrame): DataFrame containing cell properties.

    Returns:
        List[int]: List of labels that exceed the upper volume threshold.
    """
    if cell_properties_df.empty:
        print("No valid cell properties data available for applying volume threshold.")
        return image, []

    mean_volume = cell_properties_df['volume'].mean()
    std_volume = cell_properties_df['volume'].std()
    upper_volume_cutoff = mean_volume + 2 * std_volume
    print(f"Upper volume cutoff: {upper_volume_cutoff}")

    labels_above_threshold = cell_properties_df[cell_properties_df['volume'] > upper_volume_cutoff]['label'].tolist()
    print(f"Applied volume threshold: {upper_volume_cutoff}, detected {len(labels_above_threshold)} labels")
    return labels_above_threshold

def track_label_changes(original_image: np.ndarray, dummy_labeled_image: np.ndarray, relabeled_image: np.ndarray) -> pd.DataFrame:
    """
    Track changes in label values across three versions of an image via center of mass.

    Args:
        original_image (np.ndarray): The original image.
        dummy_labeled_image (np.ndarray): The image after dummy labeling.
        relabeled_image (np.ndarray): The image after relabeling.

    Returns:
        pd.DataFrame: DataFrame tracking label changes across the images.
    """
    region_props = regionprops(original_image)
    tracking_data = []
    for prop in region_props:
        centroid = tuple(map(int, prop.centroid))
        label_after_dummy = dummy_labeled_image[centroid]
        label_after_relabeling = relabeled_image[centroid]
        tracking_data.append((prop.label, label_after_dummy, label_after_relabeling))
    df = pd.DataFrame(tracking_data, columns=['Original label', 'After dummy labeling', 'After relabeling'])
    return df

def calculate_surface_area(row: pd.Series) -> pd.Series:
    """
    Calculate the surface area of a labeled region in a 3D image.

    Args:
        row (pd.Series): Row containing intensity image data.

    Returns:
        pd.Series: Row with an additional 'surface' field containing the calculated surface area.
    """
    row['surface'] = np.sum(find_boundaries(np.pad(row['intensity_image'], (1, 1), 'constant', constant_values=0), connectivity=1, mode='outer', background=0))
    return row

def calculate_distance_from_center(row: pd.Series, spheroid_center_of_mass: np.ndarray) -> pd.Series:
    """
    Calculate the distance from the centroid of a labeled region to the center of mass of the spheroid.

    Args:
        row (pd.Series): Row containing centroid data.
        spheroid_center_of_mass (np.ndarray): Array representing the center of mass of the spheroid.

    Returns:
        pd.Series: Row with additional 'distance_to_center' and 'location' fields.
    """
    row['centroid (z, y, x)'] = np.array(row['centroid (z, y, x)'])
    distance_to_center = distance.euclidean(row['centroid (z, y, x)'], spheroid_center_of_mass)
    row['distance_to_center'] = distance_to_center
    spheroid_radius = distance_to_center
    inner_radius = spheroid_radius * (1 / 3)
    middle_radius = spheroid_radius * (2 / 3)
    if distance_to_center < inner_radius:
        row['location'] = 'inner'
    elif distance_to_center < middle_radius:
        row['location'] = 'middle'
    else:
        row['location'] = 'outer'
    return row

def calculate_intensity_measurements(marker: np.ndarray, channel_name: str, relabeled_image_without_small_cells: np.ndarray) -> pd.DataFrame:
    """
    Calculate intensity measurements for a given marker channel in a 3D image.

    Args:
        marker (np.ndarray): Intensity image for the marker channel.
        channel_name (str): Name of the channel for labeling columns.
        relabeled_image_without_small_cells (np.ndarray): Labeled image without small cells.

    Returns:
        pd.DataFrame: DataFrame containing intensity measurements for each labeled region.
    """
    properties = ['label', 'inertia_tensor_eigvals', 'intensity_max', 'intensity_mean', 'intensity_min']
    intensity_measurements = pd.DataFrame(regionprops_table(relabeled_image_without_small_cells, intensity_image=marker, properties=properties))
    intensity_measurements[f'{channel_name}_inertia_tensor_eigvals'] = list(zip(intensity_measurements['inertia_tensor_eigvals-0'], intensity_measurements['inertia_tensor_eigvals-1'], intensity_measurements['inertia_tensor_eigvals-2']))
    intensity_measurements[f'{channel_name}_max_intensity'] = intensity_measurements['intensity_max']
    intensity_measurements[f'{channel_name}_mean_intensity'] = intensity_measurements['intensity_mean']
    intensity_measurements[f'{channel_name}_min_intensity'] = intensity_measurements['intensity_min']
    intensity_measurements.drop(['inertia_tensor_eigvals-0', 'inertia_tensor_eigvals-1', 'inertia_tensor_eigvals-2', 'intensity_max', 'intensity_min', 'intensity_mean'], axis=1, inplace=True)
    intensity_measurements.set_index('label', inplace=True)
    return intensity_measurements

def calculate_sphericity_ellipsoidal(row: pd.Series) -> pd.Series:
    """
    Calculate the sphericity of a labeled region assuming an ellipsoidal shape.

    Args:
        row (pd.Series): Row containing axis length data.

    Returns:
        pd.Series: Row with an additional 'sphericity_ellipsoidal' field.
    """
    semi_major_axis = row['axis_major_length'] / 2
    semi_minor_axis = row['axis_minor_length'] / 2
    if semi_major_axis == semi_minor_axis:
        row['sphericity_ellipsoidal'] = 1
    else:
        row['sphericity_ellipsoidal'] = (2 * ((semi_major_axis * semi_minor_axis ** 2) ** (1 / 3))) / (semi_major_axis + (semi_minor_axis ** 2 / ((semi_major_axis ** 2 - semi_minor_axis ** 2) ** (1 / 2))) * math.log((semi_major_axis + ((semi_major_axis ** 2 - semi_minor_axis ** 2) ** (1 / 3))) / semi_minor_axis))
    return row

def process_cell_properties(relabeled_image_without_small_cells: np.ndarray, dir_path: str, file_name: str, voxel_size: Tuple[float, float, float]) -> None:
    """
    Process and calculate cell properties for a given image and save results.

    Args:
        relabeled_image_without_small_cells (np.ndarray): The relabeled image without small cells.
        dir_path (str): Directory path for file operations.
        file_name (str): Original filename of the image.
        voxel_size (Tuple[float, float, float]): Tuple specifying the voxel size in the format (z, y, x).
    """
    intensity_file_name = 'intensity_image.tif'
    intensity_image_path = os.path.join(dir_path, intensity_file_name)

    try:
        intensity_image = tif.imread(intensity_image_path)
        intensity_image = np.moveaxis(intensity_image, 1, -1)
        print(f"Loaded intensity image from {intensity_image_path}")
    except FileNotFoundError:
        print(f"Intensity image not found at {intensity_image_path}")
        return
    except Exception as e:
        print(f"Failed to load intensity image: {str(e)}")
        return

    properties = ['label', 'area', 'axis_minor_length', 'axis_major_length', 'bbox', 'centroid', 'equivalent_diameter_area', 'image', 'intensity_image']
    cell_properties = pd.DataFrame(regionprops_table(relabeled_image_without_small_cells, intensity_image=relabeled_image_without_small_cells, properties=properties))

    cell_properties.rename(columns={'area': 'volume', 'equivalent_diameter_area': 'equivalent_diameter'}, inplace=True)

    cell_properties['centroid (z, y, x)'] = list(zip(cell_properties['centroid-0'], cell_properties['centroid-1'], cell_properties['centroid-2']))
    cell_properties['bounding_box (z_min, y_min, x_min, z_max, y_max, x_max)'] = list(zip(cell_properties['bbox-0'], cell_properties['bbox-1'], cell_properties['bbox-2'], cell_properties['bbox-3'], cell_properties['bbox-4'], cell_properties['bbox-5']))
    cell_properties.rename(columns={'area': 'volume', 'equivalent_diameter_area': 'equivalent_diameter', 'image': 'single_cell_array'}, inplace=True)
    cell_properties.drop(['centroid-0', 'centroid-1', 'centroid-2', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5'], axis=1, inplace=True)
    cell_properties.set_index('label', inplace=True)

    cell_properties = cell_properties.apply(calculate_surface_area, axis=1)

    spheroid_center_of_mass = np.array((float(intensity_image.shape[0] / 2), intensity_image.shape[1] / 2, intensity_image.shape[2] / 2))
    cell_properties = cell_properties.apply(lambda row: calculate_distance_from_center(row, spheroid_center_of_mass), axis=1)

    zoom_resampling = ((voxel_size[0] / voxel_size[1]), 1, 1, 1)
    resampled_intensity_image = zoom(intensity_image, zoom=zoom_resampling, order=0)

    marker_channel_2 = resampled_intensity_image[:, :, :, -3]  # Assuming this is the 3rd last channel
    intensity_measurements_channel_2 = calculate_intensity_measurements(marker_channel_2, 'ch02', relabeled_image_without_small_cells)

    marker_channel_3 = resampled_intensity_image[:, :, :, -2]  # Assuming this is the 2nd last channel
    intensity_measurements_channel_3 = calculate_intensity_measurements(marker_channel_3, 'ch03', relabeled_image_without_small_cells)

    cell_properties = pd.concat([cell_properties, intensity_measurements_channel_2, intensity_measurements_channel_3], axis=1)

    cell_properties['ch02_integrated_density'] = cell_properties['ch02_mean_intensity'] * cell_properties['volume']
    cell_properties['ch03_integrated_density'] = cell_properties['ch03_mean_intensity'] * cell_properties['volume']
    cell_properties['compactness'] = (36 * math.pi * (cell_properties['volume']) ** 2) / (cell_properties['surface'] ** 3)
    cell_properties['sphericity'] = ((math.pi ** (1 / 3)) * ((6 * cell_properties['volume']) ** (2 / 3))) / (cell_properties['surface'])
    cell_properties['eccentricity'] = cell_properties['axis_minor_length'] / cell_properties['axis_major_length']
    cell_properties['elongation'] = cell_properties['axis_major_length'] / cell_properties['axis_minor_length']

    cell_properties = cell_properties.apply(calculate_sphericity_ellipsoidal, axis=1)

    cell_properties.drop(['single_cell_array', 'intensity_image', 'centroid (z, y, x)', 'bounding_box (z_min, y_min, x_min, z_max, y_max, x_max)'], axis=1, inplace=True)

    output_path = os.path.join(dir_path, f"{os.path.splitext(file_name)[0]}_single_cell_morphological_features.csv")
    cell_properties.to_csv(output_path, sep='\t')

    output_path_statistics = os.path.join(dir_path, f"{os.path.splitext(file_name)[0]}_single_cell_morphological_features_statistics.csv")
    statistics = cell_properties.describe()
    statistics.to_csv(output_path_statistics, sep='\t', index=False)
    print(f"Cell properties saved to {output_path}")

