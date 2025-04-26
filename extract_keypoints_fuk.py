import pickle
import numpy as np
import csv
import sys
import os
import math

# --- Configuration ---
PICKLE_FILENAME = "ai.pk"
# <<< CHANGED FILENAME: Reflects Swing Foot Stop Logic
OUTPUT_CSV_FILENAME = "hybrik_keypoints_global_swingstop_fk_grounded.csv"

# Key in the pickle dictionary containing the RELATIVE 3D joint data
KEYPOINT_DATA_KEY = 'pred_xyz_29'
# Optional: Key for image paths
IMG_PATH_KEY = 'img_path'

# --- USER INPUT: Estimate the person's real height in meters ---
ESTIMATED_REAL_HEIGHT_METERS = 1.7 # <<< ADJUST THIS

# --- Foot Plant Configuration ---
STANCE_FOOT_JOINTS = ['l_foot', 'r_foot']
# Threshold for change in relative X of the SWING foot to consider it landed/stationary
# Needs tuning! Start potentially slightly larger than before.
SWING_X_LANDED_THRESHOLD = 0.1 # (Scaled meters per frame)

# --- Ground Adjustment (Y-Axis) ---
ADJUST_GROUND_CONTACT = True
GROUND_CONTACT_JOINTS = ['l_foot', 'r_foot'] # Ground based on max Y foot

# Indices of the desired keypoints
JOINT_INDICES = {
    'pelvis': 0,
    'l_foot': 7,
    'r_foot': 8,
    'l_hand': 20, # Optional output
    'r_hand': 21  # Optional output
}

# Order in which to write the columns in the CSV
CSV_JOINT_ORDER = ['pelvis', 'l_foot', 'r_foot', 'l_hand', 'r_hand']

# --- Helper to calculate canonical height ---
def get_canonical_height(relative_coords_frame, ref_joints_indices):
    try:
        top_idx = ref_joints_indices['top']
        bot1_idx = ref_joints_indices['bottom1']
        bot2_idx = ref_joints_indices['bottom2']
        num_joints_in_frame = relative_coords_frame.shape[0]
        if not all(idx < num_joints_in_frame for idx in [top_idx, bot1_idx, bot2_idx]):
             raise IndexError("Reference joint index out of bounds")
        top_y = relative_coords_frame[top_idx][1] # Should be 0
        bottom_y1 = relative_coords_frame[bot1_idx][1]
        bottom_y2 = relative_coords_frame[bot2_idx][1]
        max_bottom_y = max(bottom_y1, bottom_y2) # Use max Y (closest to 0)
        height = abs(top_y - max_bottom_y)
        return height + 1e-6
    except (IndexError, KeyError, TypeError) as e:
        print(f"Warning: Could not calculate canonical height: {e}.", file=sys.stderr)
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
        print(f"An unexpected error occurred loading: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(); sys.exit(1)

    # --- Validate Data ---
    # ... (Validation code remains the same) ...
    if not isinstance(data, dict) or KEYPOINT_DATA_KEY not in data: sys.exit(1)
    relative_keypoint_data = data[KEYPOINT_DATA_KEY]
    if not isinstance(relative_keypoint_data, np.ndarray) or relative_keypoint_data.ndim != 3: sys.exit(1)
    num_frames = relative_keypoint_data.shape[0]
    num_joints_total = relative_keypoint_data.shape[1]
    print(f"Found relative data for {num_frames} frames.")
    all_needed_indices_keys = set(CSV_JOINT_ORDER) | set(STANCE_FOOT_JOINTS) | set(GROUND_CONTACT_JOINTS)
    all_needed_indices = set()
    try:
        for key in all_needed_indices_keys: all_needed_indices.add(JOINT_INDICES[key])
    except KeyError as e: sys.exit(1)
    max_index_requested = max(all_needed_indices) if all_needed_indices else -1
    if max_index_requested >= num_joints_total: sys.exit(1)

    # --- Calculate Scale Factor ---
    # ... (Scale factor calculation remains the same) ...
    print("Calculating scaling factor...")
    ref_joints_for_height_names = { 'top': 'pelvis', 'bottom1': 'l_foot', 'bottom2': 'r_foot' }
    ref_joint_indices_for_height = {}
    try: ref_joint_indices_for_height = { k: JOINT_INDICES[v] for k, v in ref_joints_for_height_names.items() }
    except KeyError as e: sys.exit(1)
    canonical_height = get_canonical_height(relative_keypoint_data[0], ref_joint_indices_for_height)
    if canonical_height is None or canonical_height <= 1e-6: sys.exit(1)
    scale_factor = ESTIMATED_REAL_HEIGHT_METERS / canonical_height
    print(f"  Calculated Scale Factor: {scale_factor:.4f}")

    # Get image paths if available
    img_paths = data.get(IMG_PATH_KEY)
    if img_paths is not None and isinstance(img_paths, np.ndarray): img_paths = img_paths.tolist()

    # --- Forward Kinematics Reconstruction (Swing Foot Stop Anchor Switch) ---
    print("Reconstructing global trajectory using swing foot stop logic...")
    global_positions_all_frames = []
    scaled_relative_all_frames = [] # Store scaled relative poses

    # Pre-calculate all scaled relative coordinates
    for frame_idx in range(num_frames):
        frame_scaled_relative = {}
        for joint_name, joint_idx in JOINT_INDICES.items():
            relative_coords = relative_keypoint_data[frame_idx, joint_idx, :]
            scaled_coords = relative_coords * scale_factor
            if joint_name == 'pelvis': scaled_coords = np.array([0.0, 0.0, 0.0])
            frame_scaled_relative[joint_name] = scaled_coords
        scaled_relative_all_frames.append(frame_scaled_relative)

    # --- Initialization for Frame 0 ---
    current_global_coords_frame0 = {}
    initial_scaled_relative = scaled_relative_all_frames[0]

    # Initial anchor heuristic: Foot with smaller relative X
    l_foot_x0_rel = initial_scaled_relative['l_foot'][0]
    r_foot_x0_rel = initial_scaled_relative['r_foot'][0]
    anchor_foot_name = 'l_foot' if l_foot_x0_rel <= r_foot_x0_rel else 'r_foot'
    print(f"Initial anchor foot (heuristic): {anchor_foot_name}")

    # Initial grounding (based on Max Y / closest to 0 in initial frame)
    # ... (Initial grounding logic remains the same) ...
    y_shift_0 = 0.0
    if ADJUST_GROUND_CONTACT:
        max_y_feet_0_rel = float('-inf')
        for gc_joint in GROUND_CONTACT_JOINTS:
            foot_y_0_rel = initial_scaled_relative[gc_joint][1]
            if foot_y_0_rel > max_y_feet_0_rel: max_y_feet_0_rel = foot_y_0_rel
        y_shift_0 = -max_y_feet_0_rel
    initial_pelvis_pos_global = np.array([0.0, y_shift_0, 0.0])
    for joint_name, scaled_rel_pos in initial_scaled_relative.items():
         current_global_coords_frame0[joint_name] = initial_pelvis_pos_global + scaled_rel_pos
    global_positions_all_frames.append(current_global_coords_frame0.copy())

    # --- Loop through Frames 1 to N-1 ---
    for frame_idx in range(1, num_frames):
        prev_global_coords = global_positions_all_frames[frame_idx - 1]
        prev_scaled_relative = scaled_relative_all_frames[frame_idx - 1]
        current_scaled_relative = scaled_relative_all_frames[frame_idx]

        # anchor_foot_name is carried over from the PREVIOUS frame's decision

        anchor_pos_global_prev = prev_global_coords[anchor_foot_name]
        anchor_pos_relative_current = current_scaled_relative[anchor_foot_name]

        # --- Calculate current GLOBAL PELVIS position (X, Z based on anchor) ---
        pelvis_X_current = anchor_pos_global_prev[0] - anchor_pos_relative_current[0]
        pelvis_Z_current = anchor_pos_global_prev[2] - anchor_pos_relative_current[2]
        pelvis_Y_est = prev_global_coords['pelvis'][1] # Carry over Y for now
        # --- End Pelvis Calculation ---

        pelvis_pos_global_current_est = np.array([pelvis_X_current, pelvis_Y_est, pelvis_Z_current])

        # Calculate UNADJUSTED global positions for all joints
        current_global_coords_unadjusted = {}
        for joint_name in JOINT_INDICES.keys():
            scaled_relative_pos = current_scaled_relative[joint_name]
            current_global_coords_unadjusted[joint_name] = pelvis_pos_global_current_est + scaled_relative_pos

        # Apply Y-axis grounding for the current frame
        # ... (Y grounding logic remains the same) ...
        y_shift_current = 0.0
        current_global_coords = {}
        if ADJUST_GROUND_CONTACT:
            max_y_feet_current_unadj = float('-inf')
            for gc_joint in GROUND_CONTACT_JOINTS:
                foot_y_current_unadj = current_global_coords_unadjusted[gc_joint][1]
                if foot_y_current_unadj > max_y_feet_current_unadj: max_y_feet_current_unadj = foot_y_current_unadj
            if max_y_feet_current_unadj != float('-inf'): y_shift_current = -max_y_feet_current_unadj
            else: print(f"Warning: Could not find grounding foot Y for frame {frame_idx}", file=sys.stderr)
        shift_vector = np.array([0, y_shift_current, 0])
        for joint_name, unadj_pos in current_global_coords_unadjusted.items():
             current_global_coords[joint_name] = unadj_pos + shift_vector

        # Store the final grounded global coordinates
        global_positions_all_frames.append(current_global_coords.copy())

        # --- Determine the anchor foot for the NEXT iteration (Swing Foot Stop Logic) ---
        swing_foot_name = 'r_foot' if anchor_foot_name == 'l_foot' else 'l_foot'

        # Calculate CHANGE in relative X for the swing foot
        swing_foot_rel_x_prev = prev_scaled_relative[swing_foot_name][0]
        swing_foot_rel_x_curr = current_scaled_relative[swing_foot_name][0]
        delta_swing_rel_x = abs(swing_foot_rel_x_curr - swing_foot_rel_x_prev)

        # Decide next anchor
        next_anchor_foot_name = anchor_foot_name # Default: stick with current anchor
        if delta_swing_rel_x < SWING_X_LANDED_THRESHOLD:
            # The swing foot has slowed down enough relative to the pelvis,
            # assume it has landed and become the new anchor.
            next_anchor_foot_name = swing_foot_name
            # print(f"Frame {frame_idx}: Anchor switch! {anchor_foot_name} -> {next_anchor_foot_name} (SwingDelta:{delta_swing_rel_x:.4f} < Thresh:{SWING_X_LANDED_THRESHOLD})") # Debug

        # Update anchor foot for the next loop iteration
        anchor_foot_name = next_anchor_foot_name
        # --- End Anchor Foot Determination ---


    # --- Write results to CSV ---
    print(f"Writing reconstructed global trajectory (SwingStop FK, Grounded) to '{OUTPUT_CSV_FILENAME}'...")
    # ... (CSV writing logic remains the same) ...
    header = ['frame']
    if img_paths is not None: header.append('img_path')
    for joint_name in CSV_JOINT_ORDER: header.extend([f'{joint_name}_x', f'{joint_name}_y', f'{joint_name}_z'])
    with open(OUTPUT_CSV_FILENAME, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        for frame_idx in range(num_frames):
            row_data = [frame_idx]
            if img_paths is not None:
                try: row_data.append(os.path.basename(str(img_paths[frame_idx])))
                except Exception: row_data.append("?")
            frame_global_data = global_positions_all_frames[frame_idx]
            for joint_name in CSV_JOINT_ORDER:
                coords = frame_global_data.get(joint_name, [np.nan, np.nan, np.nan])
                row_data.extend(coords.tolist())
            csv_writer.writerow(row_data)

    print(f"Successfully saved reconstructed trajectory to '{OUTPUT_CSV_FILENAME}'.")


if __name__ == "__main__":
    main()