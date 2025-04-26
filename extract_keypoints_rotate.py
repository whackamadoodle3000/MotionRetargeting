import pickle
import numpy as np
import csv
import sys
import os
import math # Import math for trigonometric functions

# --- Configuration ---
PICKLE_FILENAME = "res_hands.pk"
# <<< CHANGED FILENAME to reflect rotation
OUTPUT_CSV_FILENAME = "hybrik_keypoints_relative_scaled_grounded_rotated.csv"

# Key in the pickle dictionary containing the RELATIVE 3D joint data
KEYPOINT_DATA_KEY = 'pred_xyz_29'
# Optional: Key for image paths
IMG_PATH_KEY = 'img_path'

# --- USER INPUT: Estimate the person's real height in meters ---
ESTIMATED_REAL_HEIGHT_METERS = 1.7 # <<< ADJUST THIS BASED ON THE PERSON IN THE VIDEO

# --- Ground Adjustment ---
# Set to True to shift coordinates vertically so the lowest foot is at Y=0
ADJUST_GROUND_CONTACT = True
# Joints considered for ground contact (must be in JOINT_INDICES below)
GROUND_CONTACT_JOINTS = ['l_foot', 'r_foot']

# --- Rotation Configuration ---
# Set to True to apply rotation
APPLY_ROTATION = True
# Rotation angle in degrees (clockwise around X-axis when looking down +X)
ROTATION_ANGLE_DEGREES = 15.0
# Rotation origin: We rotate around the pelvis *before* grounding shift

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
        # Ensure ref_joints_indices contains valid indices for the frame data
        top_idx = ref_joints_indices['top']
        bot1_idx = ref_joints_indices['bottom1']
        bot2_idx = ref_joints_indices['bottom2']

        # Check if indices are within bounds of the actual joint data array
        num_joints_in_frame = relative_coords_frame.shape[0]
        if not all(idx < num_joints_in_frame for idx in [top_idx, bot1_idx, bot2_idx]):
             raise IndexError("Reference joint index out of bounds")

        top_y = relative_coords_frame[top_idx][1] # Y-coord of top ref
        bottom_y1 = relative_coords_frame[bot1_idx][1]
        bottom_y2 = relative_coords_frame[bot2_idx][1]
        min_bottom_y = min(bottom_y1, bottom_y2) # Lowest foot Y
        height = abs(top_y - min_bottom_y) # Difference is height
        # Add small epsilon to prevent division by zero if height is somehow zero
        return height + 1e-6
    except (IndexError, KeyError, TypeError) as e: # Added TypeError for potential issues with indices
        print(f"Warning: Could not calculate canonical height due to issue: {e}. Check indices/keys.", file=sys.stderr)
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
    # Consolidate all potentially needed indices
    all_needed_indices = set(JOINT_INDICES.values())
    if ADJUST_GROUND_CONTACT:
        for gc_joint in GROUND_CONTACT_JOINTS:
            if gc_joint not in JOINT_INDICES:
                print(f"Error: Ground contact joint '{gc_joint}' is not in JOINT_INDICES.", file=sys.stderr)
                sys.exit(1)
            all_needed_indices.add(JOINT_INDICES[gc_joint])
    try:
        all_needed_indices.update({JOINT_INDICES[j_name] for j_name in CANONICAL_HEIGHT_REF_JOINTS.values()})
    except KeyError as e:
        print(f"Error: Joint name '{e}' used in CANONICAL_HEIGHT_REF_JOINTS not found in JOINT_INDICES.", file=sys.stderr)
        sys.exit(1)

    max_index_requested = max(all_needed_indices) if all_needed_indices else -1
    if max_index_requested >= num_joints_total:
        print(f"Error: Required joint index {max_index_requested} is out of bounds for {num_joints_total} joints.", file=sys.stderr)
        sys.exit(1)

    # --- Calculate Scale Factor ---
    print("Calculating scaling factor...")
    # Map joint names to indices for height calculation
    ref_joint_indices_for_height = {}
    try:
         ref_joint_indices_for_height = {
             k: JOINT_INDICES[v] for k, v in CANONICAL_HEIGHT_REF_JOINTS.items()
         }
    except KeyError as e:
         print(f"Error: Joint name '{e}' for height calc not found in JOINT_INDICES.", file=sys.stderr)
         sys.exit(1)

    # Pass the entire relative joint data for frame 0 to the height function
    canonical_height = get_canonical_height(relative_keypoint_data[0], ref_joint_indices_for_height)

    if canonical_height is None or canonical_height <= 1e-6:
        print("Error: Could not determine valid canonical height. Cannot calculate scale factor.", file=sys.stderr)
        sys.exit(1) # Exit if height is invalid

    scale_factor = ESTIMATED_REAL_HEIGHT_METERS / canonical_height
    scale_factor =1.0/0.45
    print(f"  Estimated Canonical Height (frame 0): {canonical_height:.4f} units")
    print(f"  Estimated Real Height: {ESTIMATED_REAL_HEIGHT_METERS} m")
    print(f"  Calculated Scale Factor: {scale_factor:.4f}")

    # --- Pre-calculate Rotation components ---
    cos_theta = 1.0
    sin_theta = 0.0
    if APPLY_ROTATION:
        # Clockwise rotation around X when looking down +X -> Negative angle
        angle_rad = math.radians(-ROTATION_ANGLE_DEGREES)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        print(f"Applying rotation of {ROTATION_ANGLE_DEGREES} deg clockwise around X-axis.")

    # Get image paths if available
    img_paths = data.get(IMG_PATH_KEY)
    if img_paths is not None and isinstance(img_paths, np.ndarray):
         img_paths = img_paths.tolist()

    # --- Prepare and Write Scaled, Grounded, Rotated, Relative CSV ---
    print(f"Processing frames: Scaling, Rotating relative coords, Adjusting ground contact...")
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
                    # Pelvis is the origin in the relative scaled frame (before rotation/grounding)
                    scaled_coords = np.array([0.0, 0.0, 0.0])
                else:
                    scaled_coords = relative_coords * scale_factor
                frame_scaled_relative_coords[joint_name] = scaled_coords

            # --- Apply Rotation around X-axis (around pelvis) ---
            frame_rotated_coords = {}
            if APPLY_ROTATION:
                for joint_name in JOINT_ORDER:
                    scaled_coords = frame_scaled_relative_coords[joint_name]
                    x, y, z = scaled_coords[0], scaled_coords[1], scaled_coords[2]
                    # Apply X-rotation matrix:
                    # x' = x
                    # y' = y*cos - z*sin
                    # z' = y*sin + z*cos
                    rotated_y = y * cos_theta - z * sin_theta
                    rotated_z = y * sin_theta + z * cos_theta
                    frame_rotated_coords[joint_name] = np.array([x, rotated_y, rotated_z])
            else:
                # If no rotation, just copy the scaled coordinates
                frame_rotated_coords = frame_scaled_relative_coords

            # --- Apply Ground Contact Adjustment (Optional) ---
            y_shift = 0.0 # Default: no shift
            if ADJUST_GROUND_CONTACT:
                min_y_feet = float('inf')
                for gc_joint in GROUND_CONTACT_JOINTS:
                    # Use the *rotated* coordinates for finding the minimum Y
                    try:
                        foot_y = frame_rotated_coords[gc_joint][1] # Index 1 assumed Y
                        if foot_y < min_y_feet:
                            min_y_feet = foot_y
                    except KeyError:
                         print(f"Warning: Ground contact joint '{gc_joint}' not found in processed coords for frame {frame_idx}.", file=sys.stderr)
                         # Decide how to handle: skip frame, use 0 shift, etc. Here we use 0 shift.
                         min_y_feet = 0 # Or keep it inf so y_shift remains 0
                         break # Stop checking feet for this frame if one is missing

                # Calculate shift needed IF min_y_feet was successfully found
                if min_y_feet != float('inf'):
                     y_shift = -min_y_feet


            # --- Build final row with shifted coordinates ---
            for joint_name in JOINT_ORDER:
                # Get the scaled and rotated relative coordinates
                rotated_relative_coords = frame_rotated_coords[joint_name]
                # Apply the vertical shift to ALL joints (including pelvis)
                shift_vector = np.array([0, y_shift, 0])
                final_coords = rotated_relative_coords + shift_vector
                row_data.extend(final_coords.tolist()) # Append final x, y, z

            csv_writer.writerow(row_data) # Write the complete row for the frame

    print(f"Successfully saved scaled, grounded, ROTATED, relative keypoints to '{OUTPUT_CSV_FILENAME}'.")

if __name__ == "__main__":
    main()