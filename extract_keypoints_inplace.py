import pickle
import numpy as np
import csv
import sys
import os

# --- Configuration ---
PICKLE_FILENAME = "res_hands.pk"
OUTPUT_CSV_FILENAME = "hybrik_keypoints_relative_scaled_grounded.csv" # New output name

# Key in the pickle dictionary containing the RELATIVE 3D joint data
KEYPOINT_DATA_KEY = 'pred_xyz_29'
# Optional: Key for image paths
IMG_PATH_KEY = 'img_path'

# --- USER INPUT: Estimate the person's real height in meters ---
# ESTIMATED_REAL_HEIGHT_METERS = 1.81 # <<< ADJUST THIS BASED ON THE PERSON IN THE VIDEO

ESTIMATED_REAL_HEIGHT_METERS = 1.7 # <<< ADJUST THIS BASED ON THE PERSON IN THE VIDEO

# --- Ground Adjustment ---
# Set to True to shift coordinates vertically so the lowest foot is at Y=0
ADJUST_GROUND_CONTACT = True
# Joints considered for ground contact (must be in JOINT_INDICES below)
GROUND_CONTACT_JOINTS = ['l_foot', 'r_foot']

# Indices of the desired keypoints (RELATIVE to pelvis)
JOINT_INDICES = {
    'pelvis': 0,
    'l_foot': 7,
    'r_foot': 8,
    'l_hand': 20,
    'r_hand': 21
    # Optional: Add head/neck for height calculation if needed
    # 'neck': 12,
}
# Joints used for calculating canonical height (adjust if needed)
CANONICAL_HEIGHT_REF_JOINTS = {
    'top': 'pelvis', # Using pelvis Y=0 as top reference
    'bottom1': 'l_foot',
    'bottom2': 'r_foot'
}

# Order in which to write the columns in the CSV
JOINT_ORDER = ['pelvis', 'l_foot', 'r_foot', 'l_hand', 'r_hand']


# --- Helper to calculate canonical height ---
def get_canonical_height(relative_coords_frame, ref_joints_indices):
    """Estimates height in canonical units from one frame's relative coords."""
    try:
        top_y = relative_coords_frame[ref_joints_indices['top']][1] # Y-coord of top ref
        bottom_y1 = relative_coords_frame[ref_joints_indices['bottom1']][1]
        bottom_y2 = relative_coords_frame[ref_joints_indices['bottom2']][1]
        min_bottom_y = min(bottom_y1, bottom_y2) # Lowest foot Y
        height = abs(top_y - min_bottom_y) # Difference is height
        # Add small epsilon to prevent division by zero if height is somehow zero
        return height + 1e-6
    except (IndexError, KeyError):
        print("Warning: Could not calculate canonical height due to joint index/key issue.", file=sys.stderr)
        return None

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
    # ... (add other loading error handling as before) ...
    except Exception as e:
        print(f"An unexpected error occurred loading '{PICKLE_FILENAME}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- Validate Data ---
    if not isinstance(data, dict) or KEYPOINT_DATA_KEY not in data:
        print(f"Error: Pickle data format incorrect or key '{KEYPOINT_DATA_KEY}' missing.", file=sys.stderr)
        sys.exit(1)

    relative_keypoint_data = data[KEYPOINT_DATA_KEY]
    if not isinstance(relative_keypoint_data, np.ndarray) or relative_keypoint_data.ndim < 3:
        print(f"Error: Keypoint data is not a valid numpy array.", file=sys.stderr)
        sys.exit(1)

    num_frames = relative_keypoint_data.shape[0]
    num_joints_total = relative_keypoint_data.shape[1]
    print(f"Found relative data for {num_frames} frames.")

    # Check needed joint indices exist
    all_needed_indices = set(JOINT_INDICES.values())
    if ADJUST_GROUND_CONTACT:
        for gc_joint in GROUND_CONTACT_JOINTS:
            if gc_joint not in JOINT_INDICES:
                print(f"Error: Ground contact joint '{gc_joint}' is not in JOINT_INDICES.", file=sys.stderr)
                sys.exit(1)
            all_needed_indices.add(JOINT_INDICES[gc_joint])
    all_needed_indices.update({JOINT_INDICES[j_name] for j_name in CANONICAL_HEIGHT_REF_JOINTS.values()})

    max_index_requested = max(all_needed_indices)
    if max_index_requested >= num_joints_total:
        print(f"Error: Required joint index {max_index_requested} is out of bounds for {num_joints_total} joints.", file=sys.stderr)
        sys.exit(1)

    # --- Calculate Scale Factor ---
    print("Calculating scaling factor...")
    ref_joint_indices_for_height = {
        k: JOINT_INDICES[v] for k, v in CANONICAL_HEIGHT_REF_JOINTS.items()
    }
    canonical_height = get_canonical_height(relative_keypoint_data[0], ref_joint_indices_for_height)

    if canonical_height is None or canonical_height <= 1e-6:
        print("Error: Could not determine valid canonical height. Cannot calculate scale factor.", file=sys.stderr)
        sys.exit(1) # Exit if height is invalid

    scale_factor = ESTIMATED_REAL_HEIGHT_METERS / canonical_height
    print(f"  Estimated Canonical Height (frame 0): {canonical_height:.4f} units")
    print(f"  Estimated Real Height: {ESTIMATED_REAL_HEIGHT_METERS} m")
    print(f"  Calculated Scale Factor: {scale_factor:.4f}")

    # Get image paths if available
    img_paths = data.get(IMG_PATH_KEY)
    # ... (add img_path validation as before) ...
    if img_paths is not None and isinstance(img_paths, np.ndarray):
         img_paths = img_paths.tolist()


    # --- Prepare and Write Scaled, Grounded, Relative CSV ---
    print(f"Processing frames: Scaling relative coords and adjusting ground contact...")
    print(f"Writing to '{OUTPUT_CSV_FILENAME}'...")

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

            # --- Calculate Scaled Relative Coordinates for this frame ---
            frame_scaled_relative_coords = {}
            for joint_name in JOINT_ORDER:
                joint_index = JOINT_INDICES[joint_name]
                relative_coords = relative_keypoint_data[frame_idx, joint_index, :]
                if joint_name == 'pelvis':
                    scaled_coords = np.array([0.0, 0.0, 0.0]) # Keep pelvis at origin initially
                else:
                    scaled_coords = relative_coords * scale_factor
                frame_scaled_relative_coords[joint_name] = scaled_coords

            # --- Apply Ground Contact Adjustment (Optional) ---
            y_shift = 0.0 # Default: no shift
            if ADJUST_GROUND_CONTACT:
                min_y_feet_scaled = float('inf')
                for gc_joint in GROUND_CONTACT_JOINTS:
                    # Use the *scaled* relative coordinates for finding the minimum Y
                    foot_y = frame_scaled_relative_coords[gc_joint][1] # Index 1 assumed Y
                    if foot_y < min_y_feet_scaled:
                        min_y_feet_scaled = foot_y
                # Calculate shift needed to bring the lowest scaled foot to Y=0
                y_shift = -min_y_feet_scaled

            # --- Build final row with shifted coordinates ---
            for joint_name in JOINT_ORDER:
                # Get the scaled relative coordinates
                scaled_relative_coords = frame_scaled_relative_coords[joint_name]
                # Apply the vertical shift to ALL joints (including pelvis)
                shift_vector = np.array([0, y_shift, 0])
                final_coords = scaled_relative_coords + shift_vector
                row_data.extend(final_coords.tolist()) # Append final x, y, z

            csv_writer.writerow(row_data) # Write the complete row for the frame

    print(f"Successfully saved scaled, grounded, relative keypoints to '{OUTPUT_CSV_FILENAME}'.")

if __name__ == "__main__":
    main()