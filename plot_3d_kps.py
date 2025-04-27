import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

CSV_FILE_PATH = 'hybrik_keypoints_global_gblXvel_constraint_fk_rotated.csv'
SAVE_ANIMATION = False # Set to True to save the animation as an MP4 file
OUTPUT_FILENAME = 'keypoint_animation_xz_ceiling.mp4'
ANIMATION_INTERVAL = 50  # milliseconds per frame (50ms = 20fps)
DRAW_SKELETON = True # Set to True to draw lines between keypoints

keypoints = {
    'pelvis': ('pelvis_x', 'pelvis_y', 'pelvis_z'),
    'l_foot': ('l_foot_x', 'l_foot_y', 'l_foot_z'),
    'r_foot': ('r_foot_x', 'r_foot_y', 'r_foot_z'),
    'l_hand': ('l_hand_x', 'l_hand_y', 'l_hand_z'),
    'r_hand': ('r_hand_x', 'r_hand_y', 'r_hand_z')
}

skeleton_connections = [
    ('pelvis', 'l_foot', 'blue'),
    ('pelvis', 'r_foot', 'red'),
    ('pelvis', 'l_hand', 'green'),
    ('pelvis', 'r_hand', 'orange')
]

keypoint_colors = {
    'pelvis': 'black',
    'l_foot': 'blue',
    'r_foot': 'red',
    'l_hand': 'green',
    'r_hand': 'orange'
}

try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: CSV file not found at '{CSV_FILE_PATH}'")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

required_cols = [col for kp_cols in keypoints.values() for col in kp_cols]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing required columns in CSV: {', '.join(missing_cols)}")
    exit()

print(f"Successfully loaded {len(df)} frames from '{CSV_FILE_PATH}'")

all_coords_list = []
for kp_name, cols in keypoints.items():
    all_coords_list.append(df[list(cols)].values)

all_coords_np = np.concatenate(all_coords_list, axis=0)
min_vals = np.min(all_coords_np, axis=0) 
max_vals = np.max(all_coords_np, axis=0) 
ranges = max_vals - min_vals 
padding_fraction = 0.1
padding = ranges * padding_fraction

center = (min_vals + max_vals) / 2.0
max_range = np.max(ranges) + np.max(padding) * 2 

plot_xlim = (center[0] - max_range / 2.0, center[0] + max_range / 2.0)
plot_ylim = (center[2] - max_range / 2.0, center[2] + max_range / 2.0)
min_plot_z = center[1] - max_range / 2.0
max_plot_z = center[1] + max_range / 2.0

plot_zlim_inverted = (max_plot_z, min_plot_z)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(plot_xlim)
ax.set_ylim(plot_ylim)
ax.set_zlim(plot_zlim_inverted)

ax.set_xlabel("X coordinate (Left/Right)")
ax.set_ylabel("Z coordinate (Forward/Backward)") 
ax.set_zlabel("Y coordinate (Height - Lower is Higher)")
ax.set_title("3D Keypoint Movement (XZ Ceiling Plane)")
kp_names_list = list(keypoints.keys())
num_keypoints = len(kp_names_list)

initial_x = [0] * num_keypoints
initial_plot_y = [0] * num_keypoints 
initial_plot_z = [0] * num_keypoints
colors = [keypoint_colors[kp] for kp in kp_names_list]

scatter = ax.scatter(initial_x, initial_plot_y, initial_plot_z, c=colors, marker='o', s=50)

lines = {}
if DRAW_SKELETON:
    for i, (kp1, kp2, color) in enumerate(skeleton_connections):
        line, = ax.plot([], [], [], lw=2, color=color, label=f'{kp1}-{kp2}')
        lines[(kp1, kp2)] = line


def update(frame_num):
    frame_data = df.iloc[frame_num]
    plot_xs, plot_ys, plot_zs = [], [], []
    current_kp_coords = {} 

    for kp_name in kp_names_list:
        cols = keypoints[kp_name]
        data_x, data_y, data_z = frame_data[cols[0]], frame_data[cols[1]], frame_data[cols[2]]

        plot_xs.append(data_x) # plot X <- Data X
        plot_ys.append(data_z) # plot Y <- Data Z
        plot_zs.append(data_y) # plot Z <- Data Y

        current_kp_coords[kp_name] = (data_x, data_y, data_z) 
    scatter._offsets3d = (plot_xs, plot_ys, plot_zs)

    if DRAW_SKELETON:
        for kp1, kp2, _ in skeleton_connections:
            line = lines.get((kp1, kp2))
            if line and kp1 in current_kp_coords and kp2 in current_kp_coords:
                p1_data = current_kp_coords[kp1]
                p2_data = current_kp_coords[kp2] 
                line.set_data([p1_data[0], p2_data[0]], [p1_data[2], p2_data[2]]) 
                line.set_3d_properties([p1_data[1], p2_data[1]])             

    ax.set_title(f"3D Keypoint Movement (XZ Ceiling Plane - Frame {frame_num})")

    plot_elements = [scatter]
    if DRAW_SKELETON:
        plot_elements.extend(lines.values())
    return plot_elements

num_frames = len(df)
ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=ANIMATION_INTERVAL,
    blit=False
)

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