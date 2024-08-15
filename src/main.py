# src/main.py

import os
import pandas as pd
from image_io import load_segmented_image, save_image, create_pif_file
from image_processing import (find_labels_in_few_planes, assign_dummy_label, 
                              remove_specified_labels, relabel_image, 
                              resample_image_to_isotropic_voxels)
from analysis import (calculate_statistics, apply_small_volume_threshold, 
                      identify_large_volume_labels, track_label_changes, 
                      process_cell_properties)


def main() -> None:
    """
    Main function to execute the workflow for processing segmented images.
    """
    dir_path, file_name, segmented_image = load_segmented_image()
    if segmented_image is None:
        print("No image loaded. Exiting the workflow.")
        return

    # Save the unprocessed label image for reference
    save_image(dir_path, file_name, segmented_image, '01_raw')

    # Find labels appearing in two or fewer z-planes and change them to a dummy label
    labels_in_few_planes = find_labels_in_few_planes(segmented_image)
    dummy_labeled_image = assign_dummy_label(segmented_image, labels_in_few_planes)
    save_image(dir_path, file_name, dummy_labeled_image, '02_dummy_labeled')

    # Relabel the cleaned image
    relabeled_image = relabel_image(dummy_labeled_image)
    save_image(dir_path, file_name, relabeled_image, '03_relabeled')

    # Track label changes across the process
    label_tracking_df = track_label_changes(segmented_image, dummy_labeled_image, relabeled_image)
    tracking_file_path = os.path.join(dir_path, 'label_tracking.csv')
    label_tracking_df.to_csv(tracking_file_path, index=False)
    print(f"Label tracking data saved to {tracking_file_path}")

    # Find labels still appearing in two or fewer z-planes and delete them
    labels_to_delete = find_labels_in_few_planes(relabeled_image)
    image_without_single_plane_labels = remove_specified_labels(relabeled_image, labels_to_delete)
    save_image(dir_path, file_name, image_without_single_plane_labels, '04_wo_single_plane_labels')

    # Resample the relabeled image to achieve isotropic voxel sizes
    isotropic_image = resample_image_to_isotropic_voxels(image_without_single_plane_labels, (1.9999, 0.5682, 0.5682))
    save_image(dir_path, file_name, isotropic_image, '05_resampled')

    # Calculate cell properties for the resampled image
    cell_properties_df = calculate_statistics(isotropic_image, ['label', 'area', 'axis_minor_length', 'axis_major_length', 'centroid', 'equivalent_diameter_area'],
                                              os.path.join(dir_path, "resampled_cell_properties_statistics.csv"), "Resampled cell properties")

    # Apply volume threshold for small cells and capture removed labels
    image_wo_small_cells, small_volume_cells = apply_small_volume_threshold(isotropic_image, cell_properties_df)
    save_image(dir_path, file_name, image_wo_small_cells, '06_resampled_wo_small_cells')

    # Calculate cell properties for the resampled image without small cells
    cell_properties_df_wo_small_cells = calculate_statistics(image_wo_small_cells, ['label', 'area', 'axis_minor_length', 'axis_major_length', 'centroid', 'equivalent_diameter_area'],
                                                             os.path.join(dir_path, "resampled_cell_properties_wo_small_cells_statistics.csv"), "Resampled cell properties")

    # Identify large volume labels
    large_volume_labels = identify_large_volume_labels(image_wo_small_cells, cell_properties_df_wo_small_cells)

    # Labels which were previously identified as stitching errors and which were assigned a dummy label before relabeling
    stitching_errors = label_tracking_df.loc[label_tracking_df['After dummy labeling'] == 101010, 'After relabeling'].drop_duplicates().tolist()

    """Create a list of label values to shuffle during simulation"""
    print(type(stitching_errors), type(large_volume_labels))
    merged_list = list(set(stitching_errors + large_volume_labels))
    list_wo_small_volume_cells = [cell_id for cell_id in merged_list if cell_id not in small_volume_cells]
    list_wo_single_plane_labels = [cell_id for cell_id in list_wo_small_volume_cells if cell_id not in labels_to_delete]

    print(f'Labels to shuffle before CompuCell3D simulation: {list_wo_single_plane_labels}')

    # Save removed labels to a CSV file
    labels_file_path = os.path.join(dir_path, 'list_of_cells_to_shuffle.csv')
    pd.DataFrame({'Labels': list_wo_single_plane_labels}).to_csv(labels_file_path, index=False)
    print(f"Removed labels saved to {labels_file_path}")

    # Call process_cell_properties after all other processing
    process_cell_properties(image_wo_small_cells, dir_path, file_name, (1.9999, 0.5682, 0.5682))

    # Call the function to create the PIF file
    create_pif_file(image_wo_small_cells, dir_path, file_name)


# Execute the main function
if __name__ == "__main__":
    main()
