import pickle
import numpy as np
import csv
import sys
import os

# --- Configuration ---
PICKLE_FILENAME = "ai2.pk"
# <<< CHANGED FILENAME to reflect Z adjustment
OUTPUT_CSV_FILENAME = "hybrik_keypoints_absolute_z_adjusted2.csv"

# Key in the pickle dictionary containing the RELATIVE 3D joint data
KEYPOINT_DATA_KEY = 'pred_xyz_29'
# Key for the PELVIS TRANSLATION data (absolute position of the root)
PELVIS_TRANSLATION_KEY = 'transl' # <<< TRY THIS FIRST. Adjust if needed.
# Optional: Key for image paths
IMG_PATH_KEY = 'img_path'

# --- Z-Axis Adjustment ---
# Set to True to shift coordinates so the lowest Z-value foot is at Z=0
# WARNING: This is non-standard. Grounding is usually Y=0.
ADJUST_ZERO_Z_BY_FEET = False
# Joints considered for Z=0 adjustment (must be in JOINT_INDICES below)
Z_ADJUSTMENT_JOINTS = ['l_foot', 'r_foot'] # Usually feet/ankles

# Indices of the desired keypoints within the KEYPOINT_DATA_KEY array
# Based on common SMPL 29-joint layout (0-indexed):
JOINT_INDICES = {
    'pelvis': 0,
    'l_foot': 7,
    'r_foot': 8,
    'l_hand': 20,
    'r_hand': 21
}

# Order in which to write the columns in the CSV
JOINT_ORDER = ['pelvis', 'l_foot', 'r_foot', 'l_hand', 'r_hand']


# --- Main Script ---
def main():
    print(f"Attempting to load pickle file: {PICKLE_FILENAME}")
    try:
        with open(PICKLE_FILENAME, 'rb') as f:
            data = pickle.load(f)
        print("Pickle file loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File '{PICKLE_FILENAME}' not found.", file=sys.stderr)
        sys.exit(1)
    # ... (add other standard error handling: UnpicklingError, ModuleNotFoundError, etc.) ...
    except Exception as e:
        print(f"An unexpected error occurred loading '{PICKLE_FILENAME}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Validate Data Structure ---
    if not isinstance(data, dict):
        print(f"Error: Expected data in '{PICKLE_FILENAME}' to be a dictionary.", file=sys.stderr)
        sys.exit(1)

    if KEYPOINT_DATA_KEY not in data:
        print(f"Error: Key '{KEYPOINT_DATA_KEY}' not found.", file=sys.stderr)
        print(f"Available keys: {list(data.keys())}", file=sys.stderr)
        sys.exit(1)

    if PELVIS_TRANSLATION_KEY not in data:
        print(f"Error: Pelvis translation key '{PELVIS_TRANSLATION_KEY}' not found.", file=sys.stderr)
        print(f"Available keys: {list(data.keys())}", file=sys.stderr)
        sys.exit(1)

    relative_keypoint_data = data[KEYPOINT_DATA_KEY]
    translation_data = data[PELVIS_TRANSLATION_KEY]

    # Basic validation
    # ... (add checks for numpy array, dimensions, frame count match, transl shape as before) ...
    if not isinstance(relative_keypoint_data, np.ndarray) or not isinstance(translation_data, np.ndarray):
        print(f"Error: Data for keypoints or translation is not a numpy array.", file=sys.stderr)
        sys.exit(1)
    if relative_keypoint_data.ndim < 3 or translation_data.ndim < 2:
         print(f"Error: Unexpected array dimensions.", file=sys.stderr)
         sys.exit(1)
    if relative_keypoint_data.shape[0] != translation_data.shape[0]:
        print(f"Error: Frame count mismatch between keypoints ({relative_keypoint_data.shape[0]}) and translation ({translation_data.shape[0]}).", file=sys.stderr)
        sys.exit(1)
    if translation_data.shape[-1] != 3:
         print(f"Error: Translation data ('{PELVIS_TRANSLATION_KEY}') does not seem to have 3 coords (x,y,z). Shape: {translation_data.shape}", file=sys.stderr)
         sys.exit(1)


    # --- Extract Data ---
    try:
        num_frames = relative_keypoint_data.shape[0]
        num_joints_total = relative_keypoint_data.shape[1]
        print(f"Found data for {num_frames} frames.")

        # Check joint indices validity
        max_index_requested = max(JOINT_INDICES.values())
        if max_index_requested >= num_joints_total:
            print(f"Error: Requested joint index {max_index_requested} is out of bounds.", file=sys.stderr)
            sys.exit(1)

        # Validate Z adjustment joints are requested
        if ADJUST_ZERO_Z_BY_FEET:
            for z_joint in Z_ADJUSTMENT_JOINTS:
                if z_joint not in JOINT_INDICES:
                    print(f"Error: Z-adjustment joint '{z_joint}' is not in JOINT_INDICES.", file=sys.stderr)
                    sys.exit(1)

        # Get image paths if available
        img_paths = data.get(IMG_PATH_KEY)
        if img_paths is not None and isinstance(img_paths, np.ndarray):
             img_paths = img_paths.tolist()


        # --- Prepare and Write CSV ---
        # <<< Updated print message
        print(f"Calculating absolute coordinates, adjusting Z-axis by feet, and writing to '{OUTPUT_CSV_FILENAME}'...")

        header = ['frame']
        if img_paths is not None:
            header.append('img_path')
        for joint_name in JOINT_ORDER:
            header.extend([f'{joint_name}_x', f'{joint_name}_y', f'{joint_name}_z'])

        with open(OUTPUT_CSV_FILENAME, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)

            # Iterate through each frame
            for frame_idx in range(num_frames):
                row_data = [frame_idx]

                if img_paths is not None:
                    try: row_data.append(os.path.basename(str(img_paths[frame_idx])))
                    except Exception: row_data.append("?")

                # --- Calculate Absolute Coordinates for this frame ---
                frame_translation = translation_data[frame_idx] # Shape (3,)
                frame_abs_coords = {} # Store absolute coords temporarily
                for joint_name in JOINT_ORDER:
                    joint_index = JOINT_INDICES[joint_name]
                    relative_coords = relative_keypoint_data[frame_idx, joint_index, :]
                    absolute_coords = relative_coords + frame_translation
                    frame_abs_coords[joint_name] = absolute_coords

                # --- Apply Z-Axis Adjustment (Optional) --- # <<< MODIFIED SECTION
                z_shift = 0.0 # Default: no shift
                if ADJUST_ZERO_Z_BY_FEET:
                    # Find the minimum Z among the specified Z adjustment joints
                    min_z_feet = float('inf')
                    for z_joint in Z_ADJUSTMENT_JOINTS:
                        # <<< Use index 2 for Z coordinate
                        foot_z = frame_abs_coords[z_joint][2]
                        if foot_z < min_z_feet:
                            min_z_feet = foot_z
                    # Calculate shift needed to bring the minimum Z foot to Z=0
                    # We add this shift to all Z coordinates.
                    z_shift = -min_z_feet # <<< Calculate z_shift

                # --- Build final row with adjusted coordinates ---
                for joint_name in JOINT_ORDER:
                    # Get the pre-calculated absolute coordinates
                    absolute_coords = frame_abs_coords[joint_name]
                    # Apply the Z shift (if any)
                    # Create shift vector [0, 0, z_shift] <<< MODIFIED
                    shift_vector = np.array([0, 0, z_shift])
                    final_coords = absolute_coords + shift_vector

                    row_data.extend(final_coords.tolist()) # Append final x, y, z

                csv_writer.writerow(row_data) # Write the complete row for the frame

        # <<< Updated print message
        print(f"Successfully saved absolute, Z-adjusted keypoints to '{OUTPUT_CSV_FILENAME}'.")

    # ... (add standard error handling: IndexError, KeyError, etc.) ...
    except Exception as e:
        print(f"An unexpected error occurred during processing/writing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()