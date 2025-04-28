import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import csv
import time
import os
from tqdm import tqdm

class DigitRobotParams:
    def __init__(self):
        self.pelvis_height = 1.0
        self.leg_length = 0.8
        self.arm_length = 0.6

        self.hip_yaw_limits = (-0.5, 0.5)
        self.hip_roll_limits = (-0.3, 0.3)
        self.hip_pitch_limits = (-0.9, 1.5)
        self.knee_limits = (0.0, 2.0)
        self.ankle_pitch_limits = (-0.9, 0.9)
        self.ankle_roll_limits = (-0.3, 0.3)

        self.shoulder_yaw_limits = (-1.5, 1.5)
        self.shoulder_roll_limits = (-1.0, 1.0)
        self.shoulder_pitch_limits = (-1.0, 1.0)
        self.elbow_limits = (0.0, 2.0)

        self.scale_factor = 1.0 

        self.torso_length = 0.5
        self.upper_leg_length = 0.4
        self.lower_leg_length = 0.4
        self.foot_length = 0.1
        self.upper_arm_length = 0.3
        self.lower_arm_length = 0.3

class IKBasedRetargeting:
    def __init__(self, robot_params):
        self.robot = robot_params

    def load_keypoints(self, csv_file):
        """Load keypoints from CSV file"""
        print(f"Loading keypoints from {csv_file}")
        df = pd.read_csv(csv_file)
        return df

    def preprocess_keypoints(self, df):
        """Preprocess keypoints for IK retargeting"""
        position_columns = [col for col in df.columns if any(suffix in col for suffix in ['_x', '_y', '_z'])]
        positions_df = df[position_columns]
        scaled_positions = positions_df * self.robot.scale_factor

        normalized_positions = pd.DataFrame()
        keypoints = ['pelvis', 'l_foot', 'r_foot', 'l_hand', 'r_hand']

        for keypoint in keypoints:
            if keypoint == 'pelvis':
                normalized_positions[f'{keypoint}_x'] = scaled_positions[f'{keypoint}_x']
                normalized_positions[f'{keypoint}_y'] = scaled_positions[f'{keypoint}_y']
                normalized_positions[f'{keypoint}_z'] = scaled_positions[f'{keypoint}_z']
            else:
                normalized_positions[f'{keypoint}_x'] = scaled_positions[f'{keypoint}_x'] - scaled_positions['pelvis_x']
                normalized_positions[f'{keypoint}_y'] = scaled_positions[f'{keypoint}_y'] - scaled_positions['pelvis_y']
                normalized_positions[f'{keypoint}_z'] = scaled_positions[f'{keypoint}_z'] - scaled_positions['pelvis_z']

        return normalized_positions

    def solve_inverse_kinematics(self, keypoints_df):
        """
        Solve inverse kinematics to map keypoints to robot joint angles

        Args:
            keypoints_df: DataFrame with normalized keypoint positions

        Returns:
            DataFrame with robot joint angles
        """
        print("Solving inverse kinematics...")
        joint_angles = pd.DataFrame()

        for i in tqdm(range(len(keypoints_df))):
            row = keypoints_df.iloc[i]
            pelvis_pos = np.array([row['pelvis_x'], row['pelvis_y'], row['pelvis_z']])
            l_foot_pos = np.array([row['l_foot_x'], row['l_foot_y'], row['l_foot_z']])
            r_foot_pos = np.array([row['r_foot_x'], row['r_foot_y'], row['r_foot_z']])
            l_hand_pos = np.array([row['l_hand_x'], row['l_hand_y'], row['l_hand_z']])
            r_hand_pos = np.array([row['r_hand_x'], row['r_hand_y'], row['r_hand_z']])

            l_leg_angles = self._solve_leg_ik(l_foot_pos)
            r_leg_angles = self._solve_leg_ik(r_foot_pos)

            l_arm_angles = self._solve_arm_ik(l_hand_pos)
            r_arm_angles = self._solve_arm_ik(r_hand_pos)

            pelvis_orientation = self._calculate_pelvis_orientation(l_foot_pos, r_foot_pos)

            frame_angles = {
                'pelvis_height': pelvis_pos[2],
                'pelvis_roll': pelvis_orientation[0],
                'pelvis_pitch': pelvis_orientation[1],
                'pelvis_yaw': pelvis_orientation[2],

                'l_hip_yaw': l_leg_angles[0],
                'l_hip_roll': l_leg_angles[1],
                'l_hip_pitch': l_leg_angles[2],
                'l_knee': l_leg_angles[3],
                'l_ankle_pitch': l_leg_angles[4],
                'l_ankle_roll': l_leg_angles[5],

                'r_hip_yaw': r_leg_angles[0],
                'r_hip_roll': r_leg_angles[1],
                'r_hip_pitch': r_leg_angles[2],
                'r_knee': r_leg_angles[3],
                'r_ankle_pitch': r_leg_angles[4],
                'r_ankle_roll': r_leg_angles[5],

                'l_shoulder_yaw': l_arm_angles[0],
                'l_shoulder_roll': l_arm_angles[1],
                'l_shoulder_pitch': l_arm_angles[2],
                'l_elbow': l_arm_angles[3],

                'r_shoulder_yaw': r_arm_angles[0],
                'r_shoulder_roll': r_arm_angles[1],
                'r_shoulder_pitch': r_arm_angles[2],
                'r_elbow': r_arm_angles[3]
            }

            frame_angles = self._apply_joint_limits(frame_angles)

            joint_angles = pd.concat([joint_angles, pd.DataFrame([frame_angles])], ignore_index=True)

        return joint_angles

    def _solve_leg_ik(self, foot_position):
        """Solve IK for leg given foot position relative to pelvis"""

        x, y, z = foot_position

        leg_length = np.sqrt(x**2 + y**2 + z**2)

        max_leg_length = self.robot.upper_leg_length + self.robot.lower_leg_length
        leg_length = min(leg_length, max_leg_length * 0.99)

        hip_yaw = np.arctan2(y, x)

        sagittal_distance = np.sqrt(x**2 + y**2)

        hip_roll = np.arctan2(y, z)

        c = (self.robot.upper_leg_length**2 + self.robot.lower_leg_length**2 - leg_length**2) / (2 * self.robot.upper_leg_length * self.robot.lower_leg_length)
        c = np.clip(c, -1.0, 1.0)
        knee_angle = np.pi - np.arccos(c)

        hip_pitch = np.arctan2(sagittal_distance, -z) - np.arctan2(self.robot.lower_leg_length * np.sin(knee_angle),
                                                                  self.robot.upper_leg_length + self.robot.lower_leg_length * np.cos(knee_angle))
        
        ankle_pitch = -hip_pitch - knee_angle
        ankle_roll = -hip_roll

        return np.array([hip_yaw, hip_roll, hip_pitch, knee_angle, ankle_pitch, ankle_roll])

    def _solve_arm_ik(self, hand_position):
        """Solve IK for arm given hand position relative to pelvis"""

        x, y, z = hand_position

        shoulder_offset_x = 0.0
        shoulder_offset_y = 0.2 * np.sign(y) 
        shoulder_offset_z = self.robot.torso_length

        x -= shoulder_offset_x
        y -= shoulder_offset_y
        z -= shoulder_offset_z

        arm_length = np.sqrt(x**2 + y**2 + z**2)

        max_arm_length = self.robot.upper_arm_length + self.robot.lower_arm_length
        arm_length = min(arm_length, max_arm_length * 0.99)

        shoulder_yaw = np.arctan2(y, x)

        vertical_distance = np.sqrt(x**2 + y**2)

        shoulder_roll = np.arctan2(y, x)

        c = (self.robot.upper_arm_length**2 + self.robot.lower_arm_length**2 - arm_length**2) / (2 * self.robot.upper_arm_length * self.robot.lower_arm_length)
        c = np.clip(c, -1.0, 1.0)  
        elbow_angle = np.pi - np.arccos(c)

        shoulder_pitch = np.arctan2(z, vertical_distance) - np.arctan2(self.robot.lower_arm_length * np.sin(elbow_angle),
                                                                     self.robot.upper_arm_length + self.robot.lower_arm_length * np.cos(elbow_angle))

        return np.array([shoulder_yaw, shoulder_roll, shoulder_pitch, elbow_angle])

    def _calculate_pelvis_orientation(self, l_foot_pos, r_foot_pos):
        """Calculate pelvis orientation based on feet positions"""
        feet_center = (l_foot_pos + r_foot_pos) / 2

        roll = np.arctan2(r_foot_pos[1] - l_foot_pos[1], r_foot_pos[2] - l_foot_pos[2])

        pitch = np.arctan2(feet_center[0], -feet_center[2])

        yaw = np.arctan2(r_foot_pos[1] - l_foot_pos[1], r_foot_pos[0] - l_foot_pos[0])

        return np.array([roll, pitch, yaw])

    def _apply_joint_limits(self, angles_dict):
        """Apply joint limits to angles"""
        angles_dict['l_hip_yaw'] = np.clip(angles_dict['l_hip_yaw'], *self.robot.hip_yaw_limits)
        angles_dict['l_hip_roll'] = np.clip(angles_dict['l_hip_roll'], *self.robot.hip_roll_limits)
        angles_dict['l_hip_pitch'] = np.clip(angles_dict['l_hip_pitch'], *self.robot.hip_pitch_limits)
        angles_dict['r_hip_yaw'] = np.clip(angles_dict['r_hip_yaw'], *self.robot.hip_yaw_limits)
        angles_dict['r_hip_roll'] = np.clip(angles_dict['r_hip_roll'], *self.robot.hip_roll_limits)
        angles_dict['r_hip_pitch'] = np.clip(angles_dict['r_hip_pitch'], *self.robot.hip_pitch_limits)

        angles_dict['l_knee'] = np.clip(angles_dict['l_knee'], *self.robot.knee_limits)
        angles_dict['r_knee'] = np.clip(angles_dict['r_knee'], *self.robot.knee_limits)

        angles_dict['l_ankle_pitch'] = np.clip(angles_dict['l_ankle_pitch'], *self.robot.ankle_pitch_limits)
        angles_dict['l_ankle_roll'] = np.clip(angles_dict['l_ankle_roll'], *self.robot.ankle_roll_limits)
        angles_dict['r_ankle_pitch'] = np.clip(angles_dict['r_ankle_pitch'], *self.robot.ankle_pitch_limits)
        angles_dict['r_ankle_roll'] = np.clip(angles_dict['r_ankle_roll'], *self.robot.ankle_roll_limits)

        angles_dict['l_shoulder_yaw'] = np.clip(angles_dict['l_shoulder_yaw'], *self.robot.shoulder_yaw_limits)
        angles_dict['l_shoulder_roll'] = np.clip(angles_dict['l_shoulder_roll'], *self.robot.shoulder_roll_limits)
        angles_dict['l_shoulder_pitch'] = np.clip(angles_dict['l_shoulder_pitch'], *self.robot.shoulder_pitch_limits)
        angles_dict['r_shoulder_yaw'] = np.clip(angles_dict['r_shoulder_yaw'], *self.robot.shoulder_yaw_limits)
        angles_dict['r_shoulder_roll'] = np.clip(angles_dict['r_shoulder_roll'], *self.robot.shoulder_roll_limits)
        angles_dict['r_shoulder_pitch'] = np.clip(angles_dict['r_shoulder_pitch'], *self.robot.shoulder_pitch_limits)

        angles_dict['l_elbow'] = np.clip(angles_dict['l_elbow'], *self.robot.elbow_limits)
        angles_dict['r_elbow'] = np.clip(angles_dict['r_elbow'], *self.robot.elbow_limits)

        return angles_dict

    def forward_kinematics(self, joint_angles):
        """
        Perform forward kinematics to compute keypoint positions from joint angles

        Args:
            joint_angles: DataFrame with robot joint angles

        Returns:
            DataFrame with keypoint positions
        """
        print("Computing forward kinematics...")
        keypoint_positions = pd.DataFrame()

        for i in tqdm(range(len(joint_angles))):
            angles = joint_angles.iloc[i]

            pelvis_height = angles['pelvis_height']
            pelvis_roll = angles['pelvis_roll']
            pelvis_pitch = angles['pelvis_pitch']
            pelvis_yaw = angles['pelvis_yaw']

            pelvis_pos = np.array([0, 0, pelvis_height])

            pelvis_rot = R.from_euler('xyz', [pelvis_roll, pelvis_pitch, pelvis_yaw]).as_matrix()

            l_foot_pos = self._leg_forward_kinematics(
                pelvis_pos, pelvis_rot,
                angles['l_hip_yaw'], angles['l_hip_roll'], angles['l_hip_pitch'],
                angles['l_knee'], angles['l_ankle_pitch'], angles['l_ankle_roll'],
                left=True
            )

            r_foot_pos = self._leg_forward_kinematics(
                pelvis_pos, pelvis_rot,
                angles['r_hip_yaw'], angles['r_hip_roll'], angles['r_hip_pitch'],
                angles['r_knee'], angles['r_ankle_pitch'], angles['r_ankle_roll'],
                left=False
            )

            l_hand_pos = self._arm_forward_kinematics(
                pelvis_pos, pelvis_rot,
                angles['l_shoulder_yaw'], angles['l_shoulder_roll'], angles['l_shoulder_pitch'],
                angles['l_elbow'],
                left=True
            )

            r_hand_pos = self._arm_forward_kinematics(
                pelvis_pos, pelvis_rot,
                angles['r_shoulder_yaw'], angles['r_shoulder_roll'], angles['r_shoulder_pitch'],
                angles['r_elbow'],
                left=False
            )

            frame_positions = {
                'pelvis_x': pelvis_pos[0],
                'pelvis_y': pelvis_pos[1],
                'pelvis_z': pelvis_pos[2],
                'l_foot_x': l_foot_pos[0],
                'l_foot_y': l_foot_pos[1],
                'l_foot_z': l_foot_pos[2],
                'r_foot_x': r_foot_pos[0],
                'r_foot_y': r_foot_pos[1],
                'r_foot_z': r_foot_pos[2],
                'l_hand_x': l_hand_pos[0],
                'l_hand_y': l_hand_pos[1],
                'l_hand_z': l_hand_pos[2],
                'r_hand_x': r_hand_pos[0],
                'r_hand_y': r_hand_pos[1],
                'r_hand_z': r_hand_pos[2]
            }

            keypoint_positions = pd.concat([keypoint_positions, pd.DataFrame([frame_positions])], ignore_index=True)

        return keypoint_positions

    def _leg_forward_kinematics(self, pelvis_pos, pelvis_rot, hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll, left=True):
        """Compute foot position from joint angles"""
        hip_offset_y = 0.1 * (-1 if left else 1)  
        hip_pos = pelvis_pos + np.matmul(pelvis_rot, np.array([0, hip_offset_y, 0]))

        hip_rot = pelvis_rot.copy()

        hip_rot = np.matmul(hip_rot, R.from_euler('z', hip_yaw).as_matrix())

        hip_rot = np.matmul(hip_rot, R.from_euler('x', hip_roll).as_matrix())

        hip_rot = np.matmul(hip_rot, R.from_euler('y', hip_pitch).as_matrix())

        knee_direction = np.array([0, 0, -1]) 
        knee_pos = hip_pos + np.matmul(hip_rot, knee_direction * self.robot.upper_leg_length)

        knee_rot = hip_rot.copy()
        knee_rot = np.matmul(knee_rot, R.from_euler('y', knee).as_matrix())

        ankle_direction = np.array([0, 0, -1]) 
        ankle_pos = knee_pos + np.matmul(knee_rot, ankle_direction * self.robot.lower_leg_length)

        ankle_rot = knee_rot.copy()
        ankle_rot = np.matmul(ankle_rot, R.from_euler('y', ankle_pitch).as_matrix())
        ankle_rot = np.matmul(ankle_rot, R.from_euler('x', ankle_roll).as_matrix())

        foot_direction = np.array([self.robot.foot_length/2, 0, 0]) 
        foot_pos = ankle_pos + np.matmul(ankle_rot, foot_direction)

        return foot_pos

    def _arm_forward_kinematics(self, pelvis_pos, pelvis_rot, shoulder_yaw, shoulder_roll, shoulder_pitch, elbow, left=True):
        """Compute hand position from joint angles"""
        shoulder_offset_y = 0.2 * (-1 if left else 1)  
        shoulder_pos = pelvis_pos + np.matmul(pelvis_rot, np.array([0, shoulder_offset_y, self.robot.torso_length]))

        shoulder_rot = pelvis_rot.copy()

        shoulder_rot = np.matmul(shoulder_rot, R.from_euler('z', shoulder_yaw).as_matrix())

        shoulder_rot = np.matmul(shoulder_rot, R.from_euler('x', shoulder_roll).as_matrix())

        shoulder_rot = np.matmul(shoulder_rot, R.from_euler('y', shoulder_pitch).as_matrix())

        elbow_direction = np.array([0, 0, -1]) 
        elbow_pos = shoulder_pos + np.matmul(shoulder_rot, elbow_direction * self.robot.upper_arm_length)

        elbow_rot = shoulder_rot.copy()
        elbow_rot = np.matmul(elbow_rot, R.from_euler('y', elbow).as_matrix())

        hand_direction = np.array([0, 0, -1])  
        hand_pos = elbow_pos + np.matmul(elbow_rot, hand_direction * self.robot.lower_arm_length)

        return hand_pos

    def smooth_trajectories(self, joint_angles, window_size=5):
        """Apply smoothing to joint angle trajectories"""
        print("Smoothing trajectories...")

        smoothed_angles = joint_angles.copy()

        for column in smoothed_angles.columns:
            smoothed_angles[column] = smoothed_angles[column].rolling(window=window_size, center=True).mean()

        for column in smoothed_angles.columns:
            smoothed_angles[column] = smoothed_angles[column].fillna(method='bfill').fillna(method='ffill')

        return smoothed_angles

    def evaluate_tracking_error(self, original_keypoints, retargeted_keypoints):
        """
        Calculate the tracking error between original and retargeted keypoints

        Args:
            original_keypoints: DataFrame with original keypoint positions
            retargeted_keypoints: DataFrame with retargeted keypoint positions

        Returns:
            Dict with RMSE values for each keypoint
        """
        print("Evaluating tracking error...")
        keypoints = ['pelvis', 'l_foot', 'r_foot', 'l_hand', 'r_hand']
        errors = {}

        for keypoint in keypoints:
            original_coords = np.array([
                original_keypoints[f'{keypoint}_x'],
                original_keypoints[f'{keypoint}_y'],
                original_keypoints[f'{keypoint}_z']
            ]).T

            retargeted_coords = np.array([
                retargeted_keypoints[f'{keypoint}_x'],
                retargeted_keypoints[f'{keypoint}_y'],
                retargeted_keypoints[f'{keypoint}_z']
            ]).T

            distances = np.sqrt(np.sum((original_coords - retargeted_coords)**2, axis=1))

            rmse = np.sqrt(np.mean(distances**2))
            errors[keypoint] = rmse

        all_original_coords = np.array([])
        all_retargeted_coords = np.array([])

        for keypoint in keypoints:
            original_coords = np.array([
                original_keypoints[f'{keypoint}_x'],
                original_keypoints[f'{keypoint}_y'],
                original_keypoints[f'{keypoint}_z']
            ]).T

            retargeted_coords = np.array([
                retargeted_keypoints[f'{keypoint}_x'],
                retargeted_keypoints[f'{keypoint}_y'],
                retargeted_keypoints[f'{keypoint}_z']
            ]).T

            if len(all_original_coords) == 0:
                all_original_coords = original_coords
                all_retargeted_coords = retargeted_coords
            else:
                all_original_coords = np.vstack((all_original_coords, original_coords))
                all_retargeted_coords = np.vstack((all_retargeted_coords, retargeted_coords))

        overall_distances = np.sqrt(np.sum((all_original_coords - all_retargeted_coords)**2, axis=1))
        overall_rmse = np.sqrt(np.mean(overall_distances**2))
        errors['overall'] = overall_rmse

        return errors

    def visualize_comparison(self, original_keypoints, retargeted_keypoints, output_dir, frame_rate=30):
        """
        Visualize the comparison between original and retargeted motions

        Args:
            original_keypoints: DataFrame with original keypoint positions
            retargeted_keypoints: DataFrame with retargeted keypoint positions
            output_dir: Directory to save visualizations
            frame_rate: Frame rate for visualization
        """
        print("Visualizing comparison...")
        os.makedirs(output_dir, exist_ok=True)

        connections = [
            ('pelvis', 'l_foot'),
            ('pelvis', 'r_foot'),
            ('pelvis', 'l_hand'),
            ('pelvis', 'r_hand')
        ]

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([0, 2])
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax2.set_zlim([0, 2])

        ax1.set_title('Original Human Motion')
        ax2.set_title('Retargeted Robot Motion (IK)')

        for i in tqdm(range(len(original_keypoints))):
            ax1.clear()
            ax2.clear()

            ax1.set_xlim([-1, 1])
            ax1.set_ylim([-1, 1])
            ax1.set_zlim([0, 2])
            ax2.set_xlim([-1, 1])
            ax2.set_ylim([-1, 1])
            ax2.set_zlim([0, 2])

            ax1.set_title('Original Human Motion')
            ax2.set_title('Retargeted Robot Motion (IK)')

            for keypoint in ['pelvis', 'l_foot', 'r_foot', 'l_hand', 'r_hand']:
                x = original_keypoints.iloc[i][f'{keypoint}_x']
                y = original_keypoints.iloc[i][f'{keypoint}_y']
                z = original_keypoints.iloc[i][f'{keypoint}_z']
                ax1.scatter(x, y, z, c='b', marker='o')

                ax1.text(x, y, z, keypoint)

            for start, end in connections:
                start_x = original_keypoints.iloc[i][f'{start}_x']
                start_y = original_keypoints.iloc[i][f'{start}_y']
                start_z = original_keypoints.iloc[i][f'{start}_z']

                end_x = original_keypoints.iloc[i][f'{end}_x']
                end_y = original_keypoints.iloc[i][f'{end}_y']
                end_z = original_keypoints.iloc[i][f'{end}_z']

                ax1.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 'b-')

            for keypoint in ['pelvis', 'l_foot', 'r_foot', 'l_hand', 'r_hand']:
                x = retargeted_keypoints.iloc[i][f'{keypoint}_x']
                y = retargeted_keypoints.iloc[i][f'{keypoint}_y']
                z = retargeted_keypoints.iloc[i][f'{keypoint}_z']
                ax2.scatter(x, y, z, c='r', marker='o')

                ax2.text(x, y, z, keypoint)

            for start, end in connections:
                start_x = retargeted_keypoints.iloc[i][f'{start}_x']
                start_y = retargeted_keypoints.iloc[i][f'{start}_y']
                start_z = retargeted_keypoints.iloc[i][f'{start}_z']

                end_x = retargeted_keypoints.iloc[i][f'{end}_x']
                end_y = retargeted_keypoints.iloc[i][f'{end}_y']
                end_z = retargeted_keypoints.iloc[i][f'{end}_z']

                ax2.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 'r-')

            plt.savefig(f'{output_dir}/frame_{i:04d}.png')


        plt.close()

        print(f"Visualization frames saved to {output_dir}")

    def save_joint_angles(self, joint_angles, output_file):
        """Save joint angles to CSV file"""
        print(f"Saving joint angles to {output_file}")
        joint_angles.to_csv(output_file, index=False)

    def compare_to_trajectory_optimization(self, ik_joint_angles, to_joint_angles):
        """
        Compare IK results with trajectory optimization results

        Args:
            ik_joint_angles: DataFrame with joint angles from IK
            to_joint_angles: DataFrame with joint angles from trajectory optimization

        Returns:
            Dict with comparison metrics
        """
        print("Comparing IK with trajectory optimization...")

        if set(ik_joint_angles.columns) != set(to_joint_angles.columns):
            raise ValueError("IK and TO joint angles have different columns")

        diffs = {}
        for column in ik_joint_angles.columns:
            ik_values = ik_joint_angles[column].values
            to_values = to_joint_angles[column].values

            min_length = min(len(ik_values), len(to_values))
            ik_values = ik_values[:min_length]
            to_values = to_values[:min_length]

            rmse = np.sqrt(np.mean((ik_values - to_values)**2))
            diffs[column] = rmse

        all_ik_values = np.concatenate([ik_joint_angles[column].values for column in ik_joint_angles.columns])
        all_to_values = np.concatenate([to_joint_angles[column].values for column in to_joint_angles.columns])

        min_length = min(len(all_ik_values), len(all_to_values))
        all_ik_values = all_ik_values[:min_length]
        all_to_values = all_to_values[:min_length]

        overall_rmse = np.sqrt(np.mean((all_ik_values - all_to_values)**2))
        diffs['overall'] = overall_rmse

        return diffs

    def run_pipeline(self, keypoints_file, output_dir, to_joint_angles_file=None):
        """Run the complete IK-based retargeting pipeline"""
        print(f"Running IK-based retargeting pipeline...")

        os.makedirs(output_dir, exist_ok=True)

        keypoints_df = self.load_keypoints(keypoints_file)
        normalized_keypoints = self.preprocess_keypoints(keypoints_df)

        joint_angles = self.solve_inverse_kinematics(normalized_keypoints)

        smoothed_joint_angles = self.smooth_trajectories(joint_angles)

        retargeted_keypoints = self.forward_kinematics(smoothed_joint_angles)

        tracking_errors = self.evaluate_tracking_error(normalized_keypoints, retargeted_keypoints)

        self.save_joint_angles(smoothed_joint_angles, f"{output_dir}/ik_joint_angles.csv")
        retargeted_keypoints.to_csv(f"{output_dir}/ik_retargeted_keypoints.csv", index=False)

        self.visualize_comparison(normalized_keypoints, retargeted_keypoints, f"{output_dir}/frames")

        comparison_results = None
        if to_joint_angles_file:
            to_joint_angles = pd.read_csv(to_joint_angles_file)
            comparison_results = self.compare_to_trajectory_optimization(smoothed_joint_angles, to_joint_angles)


            with open(f"{output_dir}/ik_vs_to_comparison.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Joint', 'RMSE'])
                for joint, rmse in comparison_results.items():
                    writer.writerow([joint, rmse])

        self._generate_report(output_dir, tracking_errors, comparison_results)

        return {
            'joint_angles': smoothed_joint_angles,
            'retargeted_keypoints': retargeted_keypoints,
            'tracking_errors': tracking_errors,
            'comparison_results': comparison_results
        }

    def _generate_report(self, output_dir, tracking_errors, comparison_results):
        """Generate a report with results"""
        print("Generating report...")

        with open(f"{output_dir}/ik_retargeting_report.md", 'w') as f:
            f.write("# IK-Based Motion Retargeting Report\n\n")

            f.write("## Tracking Errors (RMSE)\n\n")
            f.write("| Keypoint | RMSE |\n")
            f.write("|----------|------|\n")
            for keypoint, rmse in tracking_errors.items():
                f.write(f"| {keypoint} | {rmse:.4f} |\n")
            f.write("\n")

            if comparison_results:
                f.write("## Comparison with Trajectory Optimization (RMSE)\n\n")
                f.write("| Joint | RMSE |\n")
                f.write("|-------|------|\n")
                for joint, rmse in comparison_results.items():
                    f.write(f"| {joint} | {rmse:.4f} |\n")
                f.write("\n")

            f.write("## Conclusions\n\n")
            f.write("The IK-based retargeting approach provides a baseline for comparison with more sophisticated methods like trajectory optimization. ")
            f.write("IK is computationally efficient but may not ensure dynamic feasibility or smoothness without additional post-processing.\n\n")

            if comparison_results:
                overall_rmse = comparison_results.get('overall', 0)
                if overall_rmse > 0.5:
                    f.write("The trajectory optimization approach seems to produce significantly different results than pure IK, ")
                    f.write("suggesting that the dynamic constraints and optimization objectives have a substantial impact on the retargeted motion.\n")
                else:
                    f.write("The IK-based approach produces results relatively similar to trajectory optimization for this particular motion sequence, ")
                    f.write("suggesting that the dynamic constraints may not be a limiting factor for this specific motion.\n")


def create_video_from_frames(frames_dir, output_file, fps=30):
    """Create a video from a directory of frames"""
    import cv2

    print(f"Creating video from frames in {frames_dir}...")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])

    if not frame_files:
        print("No frames found!")
        return

    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame_file in tqdm(frame_files):
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        video.write(frame)

    video.release()

    print(f"Video saved to {output_file}")


def load_and_compare_to_trajectories(ik_keypoints_file, to_keypoints_file, output_dir):
    """
    Load and compare IK and TO keypoint trajectories

    Args:
        ik_keypoints_file: CSV file with IK keypoint positions
        to_keypoints_file: CSV file with TO keypoint positions
        output_dir: Directory to save comparison results
    """
    print(f"Comparing IK and TO keypoint trajectories...")

    os.makedirs(output_dir, exist_ok=True)

    ik_keypoints = pd.read_csv(ik_keypoints_file)
    to_keypoints = pd.read_csv(to_keypoints_file)

    keypoints = ['pelvis', 'l_foot', 'r_foot', 'l_hand', 'r_hand']
    errors = {}

    for keypoint in keypoints:
        ik_coords = np.array([
            ik_keypoints[f'{keypoint}_x'],
            ik_keypoints[f'{keypoint}_y'],
            ik_keypoints[f'{keypoint}_z']
        ]).T

        to_coords = np.array([
            to_keypoints[f'{keypoint}_x'],
            to_keypoints[f'{keypoint}_y'],
            to_keypoints[f'{keypoint}_z']
        ]).T

        min_length = min(len(ik_coords), len(to_coords))
        ik_coords = ik_coords[:min_length]
        to_coords = to_coords[:min_length]

        distances = np.sqrt(np.sum((ik_coords - to_coords)**2, axis=1))

        rmse = np.sqrt(np.mean(distances**2))
        errors[keypoint] = rmse

    all_ik_coords = np.array([])
    all_to_coords = np.array([])

    for keypoint in keypoints:
        ik_coords = np.array([
            ik_keypoints[f'{keypoint}_x'],
            ik_keypoints[f'{keypoint}_y'],
            ik_keypoints[f'{keypoint}_z']
        ]).T

        to_coords = np.array([
            to_keypoints[f'{keypoint}_x'],
            to_keypoints[f'{keypoint}_y'],
            to_keypoints[f'{keypoint}_z']
        ]).T

        min_length = min(len(ik_coords), len(to_coords))
        ik_coords = ik_coords[:min_length]
        to_coords = to_coords[:min_length]

        if len(all_ik_coords) == 0:
            all_ik_coords = ik_coords
            all_to_coords = to_coords
        else:
            all_ik_coords = np.vstack((all_ik_coords, ik_coords))
            all_to_coords = np.vstack((all_to_coords, to_coords))

    overall_distances = np.sqrt(np.sum((all_ik_coords - all_to_coords)**2, axis=1))
    overall_rmse = np.sqrt(np.mean(overall_distances**2))
    errors['overall'] = overall_rmse

    with open(f"{output_dir}/ik_vs_to_keypoints_comparison.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Keypoint', 'RMSE'])
        for keypoint, rmse in errors.items():
            writer.writerow([keypoint, rmse])

    fig, axes = plt.subplots(5, 3, figsize=(15, 20))

    for i, keypoint in enumerate(keypoints):
        for j, coord in enumerate(['x', 'y', 'z']):
            ax = axes[i, j]

            ik_values = ik_keypoints[f'{keypoint}_{coord}'].values
            to_values = to_keypoints[f'{keypoint}_{coord}'].values

            min_length = min(len(ik_values), len(to_values))
            ik_values = ik_values[:min_length]
            to_values = to_values[:min_length]

            ax.plot(range(len(ik_values)), ik_values, 'b-', label='IK')
            ax.plot(range(len(to_values)), to_values, 'r-', label='TO')

            ax.set_title(f'{keypoint} {coord}')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Position')
            ax.legend()

    plt.tight_layout()

    plt.savefig(f"{output_dir}/ik_vs_to_keypoints_comparison.png")
    plt.close()

    return errors


def main():
    """Main function to run the baseline comparison"""
    import os


    keypoints_file = "hybrik_keypoints_global_gblXvel_constraint_fk_rotated.csv"  # Required, Give the absolute file path of the csv file containing the keypoints
    to_joint_angles_file = None  # Optional, set to None if not used
    to_keypoints_file = None  # Optional, set to None if not used
    output_dir = "ik_baseline_output"  # Default output directory

    os.makedirs(output_dir, exist_ok=True)

    robot_params = DigitRobotParams()

    retargeting = IKBasedRetargeting(robot_params)

    results = retargeting.run_pipeline(keypoints_file, output_dir, to_joint_angles_file)

    if to_keypoints_file:
        load_and_compare_to_trajectories(f"{output_dir}/ik_retargeted_keypoints.csv",
                                        to_keypoints_file, output_dir)

    try:
        create_video_from_frames(f"{output_dir}/frames", f"{output_dir}/ik_retargeting_visualization.mp4")
    except ImportError:
        print("OpenCV not found. Skipping video creation.")

    print("Baseline comparison completed!")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
