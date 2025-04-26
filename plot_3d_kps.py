import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

# --- Configuration ---
CSV_FILE_PATH = 'hybrik_keypoints_global_swingstop_fk_grounded.csv'  # <--- CHANGE THIS TO YOUR CSV FILE NAME
SAVE_ANIMATION = False # Set to True to save the animation as an MP4 file
OUTPUT_FILENAME = 'keypoint_animation_xz_ceiling.mp4'
ANIMATION_INTERVAL = 50  # milliseconds per frame (e.g., 50ms = 20fps)
DRAW_SKELETON = True # Set to True to draw lines between keypoints

# Define the keypoints and their column names (remains the same)
keypoints = {
    'pelvis': ('pelvis_x', 'pelvis_y', 'pelvis_z'),
    'l_foot': ('l_foot_x', 'l_foot_y', 'l_foot_z'),
    'r_foot': ('r_foot_x', 'r_foot_y', 'r_foot_z'),
    'l_hand': ('l_hand_x', 'l_hand_y', 'l_hand_z'),
    'r_hand': ('r_hand_x', 'r_hand_y', 'r_hand_z')
}

# Define connections for the skeleton (remains the same)
skeleton_connections = [
    ('pelvis', 'l_foot', 'blue'),
    ('pelvis', 'r_foot', 'red'),
    ('pelvis', 'l_hand', 'green'),
    ('pelvis', 'r_hand', 'orange')
]

# Define colors for each keypoint (remains the same)
keypoint_colors = {
    'pelvis': 'black',
    'l_foot': 'blue',
    'r_foot': 'red',
    'l_hand': 'green',
    'r_hand': 'orange'
}
# --- End Configuration ---

# 1. Read the CSV data
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: CSV file not found at '{CSV_FILE_PATH}'")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Check if required columns exist
required_cols = [col for kp_cols in keypoints.values() for col in kp_cols]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing required columns in CSV: {', '.join(missing_cols)}")
    exit()

print(f"Successfully loaded {len(df)} frames from '{CSV_FILE_PATH}'")

# Extract all coordinate data for calculating plot limits
all_coords_list = []
for kp_name, cols in keypoints.items():
    all_coords_list.append(df[list(cols)].values)

all_coords_np = np.concatenate(all_coords_list, axis=0) # Shape (N*num_kps, 3), columns are data_x, data_y, data_z

# Calculate limits based on original data columns
min_vals = np.min(all_coords_np, axis=0) # (min_x, min_y, min_z)
max_vals = np.max(all_coords_np, axis=0) # (max_x, max_y, max_z)
ranges = max_vals - min_vals         # (range_x, range_y, range_z)

# Add padding
padding_fraction = 0.1
padding = ranges * padding_fraction

# Determine the maximum range to make the plot somewhat cubic
center = (min_vals + max_vals) / 2.0 # (center_x, center_y, center_z)
max_range = np.max(ranges) + np.max(padding) * 2 # Use max range including padding

# --- Define plot limits based on desired axis mapping ---
# Plot X axis <- Data X
plot_xlim = (center[0] - max_range / 2.0, center[0] + max_range / 2.0)
# Plot Y axis <- Data Z (forward/backward on ceiling)
plot_ylim = (center[2] - max_range / 2.0, center[2] + max_range / 2.0)
# Plot Z axis <- Data Y (vertical/height - lower values are higher up)
# Calculate the Z limits based on Data Y
min_plot_z = center[1] - max_range / 2.0
max_plot_z = center[1] + max_range / 2.0
# --- REVERSE the Z limits ---
plot_zlim_inverted = (max_plot_z, min_plot_z)
# ---

# 2. Set up the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Apply calculated cubic limits, using the INVERTED Z limits
ax.set_xlim(plot_xlim)
ax.set_ylim(plot_ylim)
ax.set_zlim(plot_zlim_inverted) # <--- USE INVERTED Z LIMITS HERE

# --- Update axis labels (The meaning is the same, but direction is flipped for Z) ---
ax.set_xlabel("X coordinate (Left/Right)")
ax.set_ylabel("Z coordinate (Forward/Backward)") # Plot Y is Data Z
ax.set_zlabel("Y coordinate (Height - Lower is Higher)") # Plot Z is Data Y
# ---

ax.set_title("3D Keypoint Movement (XZ Ceiling Plane)")

# Create plot objects (scatter for points, lines for skeleton)
kp_names_list = list(keypoints.keys())
num_keypoints = len(kp_names_list)

# Initialize scatter with dummy data - NOTE THE SWAPPED Y/Z (mapping remains the same)
initial_x = [0] * num_keypoints
initial_plot_y = [0] * num_keypoints # Corresponds to initial Z data
initial_plot_z = [0] * num_keypoints # Corresponds to initial Y data
colors = [keypoint_colors[kp] for kp in kp_names_list]

# Pass data in the order (PlotX, PlotY, PlotZ) which is (DataX, DataZ, DataY)
# Matplotlib will handle plotting within the inverted Z axis range
scatter = ax.scatter(initial_x, initial_plot_y, initial_plot_z, c=colors, marker='o', s=50)

lines = {}
if DRAW_SKELETON:
    for i, (kp1, kp2, color) in enumerate(skeleton_connections):
        # Initialize lines with empty data for PlotX, PlotY, PlotZ
        line, = ax.plot([], [], [], lw=2, color=color, label=f'{kp1}-{kp2}')
        lines[(kp1, kp2)] = line


# 3. Animation function (No changes needed here, mapping is correct)
def update(frame_num):
    frame_data = df.iloc[frame_num]

    # Prepare coordinates for plotting, swapping Y and Z data
    plot_xs, plot_ys, plot_zs = [], [], [] # Plot axes
    current_kp_coords = {} # Store original Data coords (x, y, z) for easy access

    for kp_name in kp_names_list:
        cols = keypoints[kp_name]
        data_x, data_y, data_z = frame_data[cols[0]], frame_data[cols[1]], frame_data[cols[2]]

        # Map data to plot axes
        plot_xs.append(data_x) # Plot X <- Data X
        plot_ys.append(data_z) # Plot Y <- Data Z
        plot_zs.append(data_y) # Plot Z <- Data Y

        current_kp_coords[kp_name] = (data_x, data_y, data_z) # Store original

    # Update scatter plot data - Arguments are (PlotX, PlotY, PlotZ)
    scatter._offsets3d = (plot_xs, plot_ys, plot_zs)

    # Update skeleton lines
    if DRAW_SKELETON:
        for kp1, kp2, _ in skeleton_connections:
            line = lines.get((kp1, kp2))
            if line and kp1 in current_kp_coords and kp2 in current_kp_coords:
                p1_data = current_kp_coords[kp1] # (data_x, data_y, data_z)
                p2_data = current_kp_coords[kp2] # (data_x, data_y, data_z)

                # Set line data using correct mapping:
                # Plot X uses Data X (index 0)
                # Plot Y uses Data Z (index 2)
                # Plot Z uses Data Y (index 1)
                line.set_data([p1_data[0], p2_data[0]], [p1_data[2], p2_data[2]]) # Set Plot X, Plot Y
                line.set_3d_properties([p1_data[1], p2_data[1]])                 # Set Plot Z

    # Update the title
    ax.set_title(f"3D Keypoint Movement (XZ Ceiling Plane - Frame {frame_num})")

    plot_elements = [scatter]
    if DRAW_SKELETON:
        plot_elements.extend(lines.values())
    return plot_elements

# Create the animation
num_frames = len(df)
ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=ANIMATION_INTERVAL,
    blit=False
)

# 4. Show or Save the animation
if SAVE_ANIMATION:
    try:
        print(f"Saving animation to '{OUTPUT_FILENAME}'...")
        writer = animation.FFMpegWriter(fps=1000 / ANIMATION_INTERVAL)
        ani.save(OUTPUT_FILENAME, writer=writer)
        print("Animation saved successfully.")
    except FileNotFoundError:
        print("\nError: 'ffmpeg' not found.")
        print("Please install ffmpeg to save the animation.")
        print("Alternatively, set SAVE_ANIMATION = False to just display the plot.")
        plt.show()
    except Exception as e:
        print(f"\nError saving animation: {e}")
        plt.show()
else:
    print("Displaying animation...")
    plt.show()