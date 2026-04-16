import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch
from matplotlib.path import Path

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8.5))
fig.patch.set_facecolor("#333333")  # Dark gray background
ax.set_facecolor("#333333")  # Axis background

# Set title (initially invisible)
title = ax.text(
    0.5,
    0.95,
    "E. coli Lineage Tree",
    fontsize=26,
    ha="center",
    va="top",
    transform=ax.transAxes,
    color="white",
    fontweight="bold",
    alpha=0,
)

# Define colors for different morphology types (brighter for cartoony effect)
colors = {
    "healthy": "#b8e986",  # Brighter green
    "divided": "#ffd700",  # Brighter orange (gold)
    "elongated": "#87cefa",  # Brighter blue (sky blue)
    "deformed": "#ff6347",  # Brighter red (tomato)
}

# Define node positions with increased horizontal spacing
node_positions = {
    1: (0.5, 0.8),
    2: (0.3, 0.68),
    3: (0.7, 0.68),
    4: (0.15, 0.55),
    5: (0.35, 0.55),
    6: (0.65, 0.55),
    7: (0.85, 0.55),
    8: (0.08, 0.42),
    9: (0.18, 0.42),
    10: (0.28, 0.42),
    11: (0.38, 0.42),
    12: (0.58, 0.42),
    13: (0.68, 0.42),
    14: (0.78, 0.42),
    15: (0.88, 0.42),
}

# Define node types
node_types = {
    1: "healthy",
    2: "elongated",
    3: "healthy",
    4: "deformed",
    5: "divided",
    6: "elongated",
    7: "healthy",
    8: "deformed",
    9: "deformed",
    10: "healthy",
    11: "healthy",
    12: "elongated",
    13: "divided",
    14: "divided",
    15: "healthy",
}

# Define parent-child relationships with order for sequential line drawing
edges = [
    (1, 2),
    (1, 3),
    (2, 4),
    (2, 5),
    (3, 6),
    (3, 7),
    (4, 8),
    (4, 9),
    (5, 10),
    (5, 11),
    (6, 12),
    (6, 13),
    (7, 14),
    (7, 15),
]

# Group nodes by level for animation
levels = [
    [1],  # Level 1: ID-1
    [2, 3],  # Level 2: ID-2, ID-3
    [4, 5, 6, 7],  # Level 3: ID-4 to ID-7
    [8, 9, 10, 11, 12, 13, 14, 15],  # Level 4: ID-8 to ID-15
]

# Track animation state for each node and edge
node_states = {
    node_id: {"frame_appeared": -1, "animation_phase": 0}
    for node_id in node_positions.keys()
}
edge_states = {(parent, child): {"frame_appeared": -1} for parent, child in edges}


# Function to create a star-like shape with subtle spikes
def create_star_shape(x, y, width, height, wobble, num_spikes=8):
    vertices = []
    codes = []
    # Calculate the step angle for the spikes
    angle_step = (
        2 * np.pi / (num_spikes * 2)
    )  # Alternate between inner and outer points
    for i in range(num_spikes * 2):
        angle = i * angle_step
        # Alternate between outer (spike) and inner (base) points
        radius = (
            (width / 2 + wobble * 0.005)
            if i % 2 == 0
            else (width / 2 - 0.01 - wobble * 0.005)
        )
        # Adjust the radius vertically to account for the height
        vert_radius = radius * (height / width)
        vert_x = x + radius * np.cos(angle)
        vert_y = y + vert_radius * np.sin(angle)
        vertices.append((vert_x, vert_y))
        codes.append(Path.MOVETO if i == 0 else Path.LINETO)
    vertices.append(vertices[0])  # Close the path
    codes.append(Path.CLOSEPOLY)
    return Path(vertices, codes)


# Animation function
def update(frame):
    # Clear previous artists (except title)
    for artist in list(ax.patches) + list(ax.lines) + list(ax.texts):
        if artist != title:
            artist.remove()

    # Fade in the title with a slight bounce
    title_alpha = min(frame / 10, 1)  # Slower fade-in over 10 frames
    title_scale = 1 + 0.1 * np.sin(np.pi * min(frame / 10, 1))  # Slower bounce effect
    title.set_alpha(title_alpha)
    title.set_fontsize(26 * title_scale)
    ax.add_artist(title)

    # Determine which levels to show based on the frame
    current_level = min(
        frame // 20, len(levels) - 1
    )  # Each level appears every 20 frames
    nodes_to_show = []
    for i in range(current_level + 1):
        nodes_to_show.extend(levels[i])

    # Update node states
    for node_id in nodes_to_show:
        if node_states[node_id]["frame_appeared"] == -1:
            node_states[node_id]["frame_appeared"] = frame
        node_states[node_id]["animation_phase"] = (
            frame - node_states[node_id]["frame_appeared"]
        )

    # Update edge states (lines appear before nodes)
    for i, (parent, child) in enumerate(edges):
        if parent in nodes_to_show and child in nodes_to_show:
            # Introduce a delay for the second line from the same parent
            delay = 5 if i % 2 == 1 else 0  # Delay the second line by 5 frames
            if edge_states[(parent, child)]["frame_appeared"] == -1:
                edge_states[(parent, child)]["frame_appeared"] = frame + delay

    # Draw edges (lines) first
    for parent, child in edges:
        if parent in nodes_to_show and child in nodes_to_show:
            edge_frame = edge_states[(parent, child)]["frame_appeared"]
            if frame >= edge_frame:
                parent_pos = node_positions[parent]
                child_pos = node_positions[child]
                edge_alpha = min(
                    (frame - edge_frame) / 10, 1
                )  # Slower fade-in over 10 frames
                ax.plot(
                    [parent_pos[0], child_pos[0]],
                    [parent_pos[1], child_pos[1]],
                    "gray",
                    linewidth=2,
                    zorder=1,
                    alpha=edge_alpha,
                )

    # Draw nodes with cartoony animations (after lines)
    for node_id in nodes_to_show:
        x, y = node_positions[node_id]
        phase = node_states[node_id]["animation_phase"]
        bounce = 1 + 0.2 * np.sin(
            np.pi * min(phase / 10, 1)
        )  # Slower bounce over 10 frames

        # Set base size based on level and cell type
        if node_id <= 7:  # Larger nodes for the first 3 levels
            base_width, base_height = 0.09, 0.04
        else:  # Smaller nodes for the last level
            base_width, base_height = 0.07, 0.03

        # Reduce size for "divided" cells
        if node_types[node_id] == "divided":
            base_width *= 0.5  # Make divided cells half the width
            base_height *= 0.5  # Make divided cells half the height

        # Apply bounce to size
        width = base_width * bounce
        height = base_height * bounce

        # Handle different cell types
        if node_types[node_id] == "healthy":
            # Pulsing effect (slower)
            pulse = 1 + 0.05 * np.sin(
                2 * np.pi * frame / 40
            )  # Slower pulsing every 40 frames
            width *= pulse
            height *= pulse
            rect = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle=f"round,pad=0,rounding_size={0.02 if node_id <= 7 else 0.015}",
                facecolor=colors[node_types[node_id]],
                edgecolor="white",
                linewidth=2,
                zorder=2,
            )
            ax.add_patch(rect)
            # Draw nucleoids for larger nodes
            if node_id <= 7:
                ellipse1 = Ellipse(
                    (x - 0.015, y),
                    0.03 * pulse,
                    0.02 * pulse,
                    facecolor="#b7b0c9",
                    edgecolor=None,
                    alpha=0.8,
                    zorder=3,
                )
                ellipse2 = Ellipse(
                    (x + 0.015, y),
                    0.03 * pulse,
                    0.02 * pulse,
                    facecolor="#b7b0c9",
                    edgecolor=None,
                    alpha=0.8,
                    zorder=3,
                )
                ax.add_patch(ellipse1)
                ax.add_patch(ellipse2)

        elif node_types[node_id] == "elongated":
            # Elongation animation (slower)
            elongation = min(phase / 10, 1)  # Slower elongation over 10 frames
            width = base_width * (1 + 0.5 * elongation)  # Stretch horizontally
            height = base_height * (
                1 - 0.2 * elongation
            )  # Slightly compress vertically
            rect = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle=f"round,pad=0,rounding_size={0.02 if node_id <= 7 else 0.015}",
                facecolor=colors[node_types[node_id]],
                edgecolor="white",
                linewidth=2,
                zorder=2,
            )
            ax.add_patch(rect)
            # Draw nucleoids for larger nodes
            if node_id <= 7:
                ellipse1 = Ellipse(
                    (x - 0.015 * (1 + 0.5 * elongation), y),
                    0.03 * (1 + 0.5 * elongation),
                    0.02,
                    facecolor="#b7b0c9",
                    edgecolor=None,
                    alpha=0.8,
                    zorder=3,
                )
                ellipse2 = Ellipse(
                    (x + 0.015 * (1 + 0.5 * elongation), y),
                    0.03 * (1 + 0.5 * elongation),
                    0.02,
                    facecolor="#b7b0c9",
                    edgecolor=None,
                    alpha=0.8,
                    zorder=3,
                )
                ax.add_patch(ellipse1)
                ax.add_patch(ellipse2)

        elif node_types[node_id] == "divided":
            # Divided cell: single smaller cell with a "pop" effect
            pop = 1 + 0.1 * np.sin(
                2 * np.pi * min(phase / 10, 1)
            )  # Pop effect over 10 frames
            width *= pop
            height *= pop
            rect = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle=f"round,pad=0,rounding_size={0.01 if node_id <= 7 else 0.0075}",
                facecolor=colors[node_types[node_id]],
                edgecolor="white",
                linewidth=1,
                zorder=2,
            )
            ax.add_patch(rect)
            # Draw nucleoids for larger nodes (adjusted for smaller size)
            if node_id <= 7:
                ellipse1 = Ellipse(
                    (x - 0.0075, y),
                    0.015 * pop,
                    0.01 * pop,
                    facecolor="#b7b0c9",
                    edgecolor=None,
                    alpha=0.8,
                    zorder=3,
                )
                ellipse2 = Ellipse(
                    (x + 0.0075, y),
                    0.015 * pop,
                    0.01 * pop,
                    facecolor="#b7b0c9",
                    edgecolor=None,
                    alpha=0.8,
                    zorder=3,
                )
                ax.add_patch(ellipse1)
                ax.add_patch(ellipse2)

        elif node_types[node_id] == "deformed":
            # Deformation animation: wobble effect with star-like edges
            wobble = 0.05 * np.sin(
                2 * np.pi * phase / 20
            )  # Slower wobble every 20 frames
            width = base_width * (1 + wobble)
            height = base_height * (1 - wobble)
            # Create a star-like shape with subtle spikes
            path = create_star_shape(x, y, width, height, wobble, num_spikes=8)
            rect = PathPatch(
                path,
                facecolor=colors[node_types[node_id]],
                edgecolor="white",
                linewidth=2,
                zorder=2,
            )
            ax.add_patch(rect)
            # Draw nucleoids for larger nodes, adjusted for the wobble
            if node_id <= 7:
                ellipse1 = Ellipse(
                    (x - 0.015, y),
                    0.03 * (1 + wobble),
                    0.02 * (1 - wobble),
                    facecolor="#b7b0c9",
                    edgecolor=None,
                    alpha=0.8,
                    zorder=3,
                )
                ellipse2 = Ellipse(
                    (x + 0.015, y),
                    0.03 * (1 + wobble),
                    0.02 * (1 - wobble),
                    facecolor="#b7b0c9",
                    edgecolor=None,
                    alpha=0.8,
                    zorder=3,
                )
                ax.add_patch(ellipse1)
                ax.add_patch(ellipse2)

        # Add node ID with slight bounce (adjust font size for smaller divided cells)
        label_y = y + (0.03 if node_id <= 7 else 0.025)
        label_fontsize = (
            (10 if node_id <= 7 else 8)
            if node_types[node_id] != "divided"
            else (8 if node_id <= 7 else 6)
        )
        label = ax.text(
            x,
            label_y,
            f"ID:{node_id}",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            zorder=4,
            color="white",
            fontweight="bold",
        )
        label.set_alpha(min(phase / 10, 1))  # Slower fade-in over 10 frames

    return []


# Animation setup
ani = FuncAnimation(fig, update, frames=120, interval=200, blit=False)

# Hide axes
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Show and save animation with error handling
plt.tight_layout()

# Try saving as MP4 first (more reliable)
try:
    print("Saving animation as MP4...")
    ani.save(
        "ecoli_lineage_tree.mp4", writer="ffmpeg", fps=5
    )  # Slower playback (5 fps)
    print("MP4 saved successfully!")
except Exception as e:
    print(f"Failed to save MP4: {e}")

# Try saving as GIF
try:
    print("Saving animation as GIF...")
    ani.save(
        "ecoli_lineage_tree.gif", writer="pillow", fps=5
    )  # Slower playback (5 fps)
    print("GIF saved successfully!")
except Exception as e:
    print(f"Failed to save GIF: {e}")

# Display the animation
plt.show()
