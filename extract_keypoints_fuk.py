import pickle
import numpy as np
import csv
import sys
import os
import math

PICKLE_FILENAME = "ai5.pk"
OUTPUT_CSV_FILENAME = "hybrik_keypoints_global_gblXvel_constraint_fk_rotated.csv"
KEYPOINT_DATA_KEY = 'pred_xyz_29'
IMG_PATH_KEY = 'img_path'
ESTIMATED_REAL_HEIGHT_METERS = 1.7 # persons height meters
APPLY_INITIAL_ROTATION = True
ROTATION_ANGLE_DEGREES = 7.5 # idk why but offset works 
STANCE_FOOT_JOINTS = ['l_foot', 'r_foot']

# threshold for change in GLOBAL X of the candidate foot to allow it to become anchor
GLOBAL_X_STATIONARY_THRESHOLD = 0.02

ADJUST_GROUND_CONTACT = True
GROUND_CONTACT_JOINTS = ['l_foot', 'r_foot']

JOINT_INDICES = {
    'pelvis': 0,
    'l_foot': 7,
    'r_foot': 8,
    'l_hand': 20, 
    'r_hand': 21 
}

CSV_JOINT_ORDER = ['pelvis', 'l_foot', 'r_foot', 'l_hand', 'r_hand']

def get_canonical_height(unrotated_relative_coords_frame, ref_joints_indices):
    try:
        top_idx = ref_joints_indices['top']
        bot1_idx = ref_joints_indices['bottom1']
        bot2_idx = ref_joints_indices['bottom2']
        num_joints_in_frame = unrotated_relative_coords_frame.shape[0]
        if not all(idx < num_joints_in_frame for idx in [top_idx, bot1_idx, bot2_idx]):
             raise IndexError("joint index out of bounds")
        top_y = unrotated_relative_coords_frame[top_idx][1]
        bottom_y1 = unrotated_relative_coords_frame[bot1_idx][1]
        bottom_y2 = unrotated_relative_coords_frame[bot2_idx][1]
        max_bottom_y = max(bottom_y1, bottom_y2)
        height = abs(top_y - max_bottom_y)
        return height + 1e-6
    except (IndexError, KeyError, TypeError) as e:
        print(f"could not calculate canonical height: {e}.", file=sys.stderr)
        return None

def main():
    print(f"attempting to load pickle file: {PICKLE_FILENAME}")
    try:
        with open(PICKLE_FILENAME, 'rb') as f:
            data = pickle.load(f)
        print("pickle file loaded successfully.")
    except FileNotFoundError:
        print(f"file '{PICKLE_FILENAME}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"error occurred loading: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(); sys.exit(1)

    if not isinstance(data, dict) or KEYPOINT_DATA_KEY not in data: sys.exit(1)
    relative_keypoint_data = data[KEYPOINT_DATA_KEY]
    if not isinstance(relative_keypoint_data, np.ndarray) or relative_keypoint_data.ndim != 3: sys.exit(1)
    num_frames = relative_keypoint_data.shape[0]
    num_joints_total = relative_keypoint_data.shape[1]
    print(f"found relative data for {num_frames} frames.")
    all_needed_indices_keys = set(CSV_JOINT_ORDER) | set(STANCE_FOOT_JOINTS) | set(GROUND_CONTACT_JOINTS)
    all_needed_indices = set()
    try:
        for key in all_needed_indices_keys: all_needed_indices.add(JOINT_INDICES[key])
    except KeyError as e: sys.exit(1)
    max_index_requested = max(all_needed_indices) if all_needed_indices else -1
    if max_index_requested >= num_joints_total: sys.exit(1)

    print("calculating scaling factor (using original orientation)...")
    ref_joints_for_height_names = { 'top': 'pelvis', 'bottom1': 'l_foot', 'bottom2': 'r_foot' }
    ref_joint_indices_for_height = {}
    try: ref_joint_indices_for_height = { k: JOINT_INDICES[v] for k, v in ref_joints_for_height_names.items() }
    except KeyError as e: sys.exit(1)
    canonical_height = get_canonical_height(relative_keypoint_data[0], ref_joint_indices_for_height)
    if canonical_height is None or canonical_height <= 1e-6: sys.exit(1)
    scale_factor = ESTIMATED_REAL_HEIGHT_METERS / canonical_height
    print(f"  calculated scale factor: {scale_factor:.4f}")

    cos_theta = 1.0
    sin_theta = 0.0
    if APPLY_INITIAL_ROTATION:
        angle_rad = math.radians(-ROTATION_ANGLE_DEGREES) #clockwise
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        print(f"applying initial rotation of {ROTATION_ANGLE_DEGREES} deg clockwise around X-axis to relative poses")


    img_paths = data.get(IMG_PATH_KEY)
    if img_paths is not None and isinstance(img_paths, np.ndarray): img_paths = img_paths.tolist()


    print("pre-processing relative data (scaling and initial rotation)...")
    processed_relative_all_frames = []
    for frame_idx in range(num_frames):
        frame_processed_relative = {}
        for joint_name, joint_idx in JOINT_INDICES.items():
            relative_coords = relative_keypoint_data[frame_idx, joint_idx, :]
            scaled_coords = relative_coords * scale_factor
            if joint_name == 'pelvis': scaled_coords = np.array([0.0, 0.0, 0.0])
            final_relative_coords = scaled_coords
            if APPLY_INITIAL_ROTATION and joint_name != 'pelvis':
                x, y, z = scaled_coords[0], scaled_coords[1], scaled_coords[2]
                rotated_y = y * cos_theta - z * sin_theta
                rotated_z = y * sin_theta + z * cos_theta
                final_relative_coords = np.array([x, rotated_y, rotated_z])
            frame_processed_relative[joint_name] = final_relative_coords
        processed_relative_all_frames.append(frame_processed_relative)


    print("reconstructing global trajectory using global X velocity constraint...")
    global_positions_all_frames = []
    current_global_coords_frame0 = {}
    initial_processed_relative = processed_relative_all_frames[0]


    l_foot_x0_rel = initial_processed_relative['l_foot'][0]
    r_foot_x0_rel = initial_processed_relative['r_foot'][0]
    anchor_foot_name = 'l_foot' if l_foot_x0_rel <= r_foot_x0_rel else 'r_foot'
    print(f"initial anchor foot (heuristic): {anchor_foot_name}")


    y_shift_0 = 0.0
    if ADJUST_GROUND_CONTACT:
        max_y_feet_0_rel = float('-inf')
        grounding_foot_0 = None
        for gc_joint in GROUND_CONTACT_JOINTS:
            foot_y_0_rel = initial_processed_relative[gc_joint][1]
            if foot_y_0_rel > max_y_feet_0_rel:
                 max_y_feet_0_rel = foot_y_0_rel
                 grounding_foot_0 = gc_joint #track which foot determined the ground
        if grounding_foot_0 is not None:
             y_shift_0 = -max_y_feet_0_rel
        else:
             print("could not determine grounding foot for frame 0", file=sys.stderr)

    initial_pelvis_pos_global = np.array([0.0, y_shift_0, 0.0])
    for joint_name, processed_rel_pos in initial_processed_relative.items():
         current_global_coords_frame0[joint_name] = initial_pelvis_pos_global + processed_rel_pos
    global_positions_all_frames.append(current_global_coords_frame0.copy())


    for frame_idx in range(1, num_frames):

        prev_global_coords = global_positions_all_frames[frame_idx - 1]
        current_processed_relative = processed_relative_all_frames[frame_idx] 
        # anchor_foot_name is the anchor used to calculate this frame's position

        anchor_pos_global_prev = prev_global_coords[anchor_foot_name]
        anchor_pos_relative_current = current_processed_relative[anchor_foot_name]
        pelvis_X_current = anchor_pos_global_prev[0] - anchor_pos_relative_current[0]
        pelvis_Z_current = anchor_pos_global_prev[2] - anchor_pos_relative_current[2]
        pelvis_Y_est = prev_global_coords['pelvis'][1] 

        pelvis_pos_global_current_est = np.array([pelvis_X_current, pelvis_Y_est, pelvis_Z_current])

        current_global_coords_unadjusted = {}
        for joint_name in JOINT_INDICES.keys():
            processed_relative_pos = current_processed_relative[joint_name]
            current_global_coords_unadjusted[joint_name] = pelvis_pos_global_current_est + processed_relative_pos

        y_shift_current = 0.0
        current_global_coords = {} 
        if ADJUST_GROUND_CONTACT:
            max_y_feet_current_unadj = float('-inf')
            grounding_foot_this_frame = None
            for gc_joint in GROUND_CONTACT_JOINTS:
                foot_y_current_unadj = current_global_coords_unadjusted[gc_joint][1]
                if foot_y_current_unadj > max_y_feet_current_unadj:
                    max_y_feet_current_unadj = foot_y_current_unadj
                    grounding_foot_this_frame = gc_joint
            if grounding_foot_this_frame is not None:
                 y_shift_current = -max_y_feet_current_unadj
            else:
                 print(f"could not find grounding foot Y for frame {frame_idx}", file=sys.stderr)

        shift_vector = np.array([0, y_shift_current, 0])
        for joint_name, unadj_pos in current_global_coords_unadjusted.items():
             current_global_coords[joint_name] = unadj_pos + shift_vector

        global_positions_all_frames.append(current_global_coords.copy())

        candidate_foot_name = 'r_foot' if anchor_foot_name == 'l_foot' else 'l_foot'
        next_anchor_foot_name = anchor_foot_name # default is stick with current anchor

        candidate_rel_y = current_processed_relative[candidate_foot_name][1]
        anchor_rel_y = current_processed_relative[anchor_foot_name][1]

        if candidate_rel_y >= anchor_rel_y: 
            candidate_global_x_curr = current_global_coords[candidate_foot_name][0]
            candidate_global_x_prev = prev_global_coords[candidate_foot_name][0]
            delta_candidate_global_x = abs(candidate_global_x_curr - candidate_global_x_prev)
            if delta_candidate_global_x < GLOBAL_X_STATIONARY_THRESHOLD:
                next_anchor_foot_name = candidate_foot_name
        anchor_foot_name = next_anchor_foot_name



    print(f"writing reconstructed global trajectory (GblXVel Constraint FK, Grounded, Rotated) to '{OUTPUT_CSV_FILENAME}'...")
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

    print(f"saved reconstructed trajectory to '{OUTPUT_CSV_FILENAME}'.")


if __name__ == "__main__":
    main()