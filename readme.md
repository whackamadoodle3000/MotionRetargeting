# Motion Retargeting 

This repository contains scripts to to retarget human motion from RGB videos to Digit robot trajectories.

## Workflow

1.  **Run HybrIK on PACE:**
    *   Follow the instructions in the  HybrIK repository's README to run the model on video data (we used the PACE cluster). This will involve setting up the environment, getting your input video (can be real or AI generated), and running the HybrIK inference script.
    *   The output you'll need from HybrIK is the `.pkl` file containing the detected keypoints and other relevant information.

2.  **Extract Keypoints:**
    *   Transfer the HybrIK output `.pkl` file to your local machine or wherever you plan to run the next step.
    *   Modify the `INPUT_PKL_PATH` and `OUTPUT_CSV_PATH` variables in `extract_keypoints_fk.py`.
    *   Run the script after install dependencies:
        ```bash
        python extract_keypoints_fk.py
        ```
    *   This script extracts the relevant 3D keypoint coordinates (pelvis, feet, hands) from the `.pkl` file. Since HybrIK has very poor depth information, it reconstructs the 3d translation using the relative motion of the legs. It will save the results to a CSV file.

3.  **Visualize Keypoints:**
    *   Ensure the generated CSV file name matches the `CSV_FILE_PATH` variable in `plot_3d_kps.py`.
    *   Run the visualization script:
        ```bash
        python plot_3d_kps.py
        ```
    *   This will display an animated 3D plot showing the movement of the extracted keypoints over time. You can see how well it tracked the motion of the person moving.


## Scripts

*   `extract_keypoints_fk.py`: Extracts and processes keypoints from HybrIK's `.pkl` output, and will create reconstructed 3d translation based on when the feet contact the ground and are stationary in the non-depth axises.
*   `plot_3d_kps.py`: Visualizes the extracted 3D keypoints from the CSV file.
*   `IK_BaselineComparison/InverseKinematics.py`: This script performs a baseline comparison using inverse kinematics (IK)-based retargeting for robot motion, given 3D keypoints as input. 
