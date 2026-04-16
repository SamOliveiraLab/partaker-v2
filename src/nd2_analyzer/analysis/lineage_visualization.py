import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.patches import FancyBboxPatch, Ellipse, PathPatch
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
import networkx as nx
import pandas as pd
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QRadioButton,
                               QComboBox, QLabel, QButtonGroup, QPushButton,
                               QMessageBox, QFileDialog, QDialogButtonBox, QApplication)
from PySide6.QtCore import Qt


class LineageVisualization:
    def __init__(self, morphology_colors_rgb=None):
        self.morphology_colors_rgb = morphology_colors_rgb or {
            "Artifact": (128/255, 128/255, 128/255),  # Gray
            "Divided": (255/255, 0/255, 0/255),       # Blue
            "Healthy": (0/255, 255/255, 0/255),       # Green
            "Elongated": (0/255, 255/255, 255/255),   # Yellow
            "Deformed": (255/255, 0/255, 255/255),    # Magenta
        }

        # Store states needed for animations
        self.node_states = {}
        self.edge_states = {}
        self.cell_objects = {}
        self.nucleoid_objects = {}
        self.frame = 0
        self.current_animation = None

    def create_lineage_tree(self, tracks, canvas, root_cell_id=None):
        """
        Create a lineage tree visualization using NetworkX on a provided canvas.

        Parameters:
        -----------
        tracks : list
            List of track dictionaries with lineage information (ID, t, children).
        canvas : FigureCanvasQTAgg
            The Matplotlib canvas to draw on.
        root_cell_id : int or None
            If provided, visualize only the lineage tree starting from this cell.
        """
        # Clear the existing figure
        canvas.figure.clear()

        import networkx as nx

        # Create directed graph
        G = nx.DiGraph()

        try:
            # Add nodes for each track
            for track in tracks:
                track_id = track['ID']

                # Get timing information
                if 't' in track and len(track['t']) > 0:
                    start_time = int(min(track['t']))
                    end_time = int(max(track['t']))
                    duration = end_time - start_time + 1
                else:
                    start_time = 0
                    end_time = 0
                    duration = 0

                # Determine if this track divides
                has_children = 'children' in track and track['children']

                # Add node with attributes
                G.add_node(track_id,
                           start_time=start_time,
                           end_time=end_time,
                           duration=duration,
                           divides=has_children)

            # Add edges for parent-child relationships
            for track in tracks:
                if 'children' in track and track['children']:
                    for child_id in track['children']:
                        G.add_edge(track['ID'], child_id)

            # Filter based on root_cell_id if provided
            if root_cell_id is not None:
                descendants = set()

                def get_descendants(node):
                    descendants.add(node)
                    if G.nodes[node]['divides']:
                        for child in G.neighbors(node):
                            get_descendants(child)
                get_descendants(root_cell_id)
                G = G.subgraph(descendants).copy()
                print(
                    f"Visualizing lineage tree for cell {root_cell_id} with {len(G.nodes())} nodes")
            else:
                # Find connected components and take top 5
                connected_components = list(nx.weakly_connected_components(G))
                print(
                    f"Found {len(connected_components)} separate lineage trees")
                largest_components = sorted(
                    connected_components, key=len, reverse=True)
                top_components = largest_components[:min(
                    5, len(largest_components))]
                G = G.subgraph(set.union(*top_components)).copy()

            # Add subplot
            if root_cell_id is None and len(top_components) > 1:
                axes = []
                for i in range(len(top_components)):
                    ax = canvas.figure.add_subplot(
                        1, len(top_components), i + 1)
                    axes.append(ax)
            else:
                ax = canvas.figure.add_subplot(111)
                axes = [ax]

            # Plot each component or single tree
            if root_cell_id is None and len(top_components) > 1:
                for i, component in enumerate(top_components):
                    subgraph = G.subgraph(component)
                    roots = [n for n in subgraph.nodes(
                    ) if subgraph.in_degree(n) == 0]
                    if not roots:
                        axes[i].text(0.5, 0.5, "No root node",
                                     ha='center', va='center')
                        axes[i].axis('off')
                        continue

                    pos = self.hierarchy_pos(subgraph, roots[0])
                    node_sizes = [100 + subgraph.nodes[n]
                                  ['duration'] * 10 for n in subgraph.nodes()]
                    node_colors = [
                        'red' if subgraph.nodes[n]['divides'] else 'blue' for n in subgraph.nodes()]

                    nx.draw_networkx_nodes(
                        subgraph,
                        pos,
                        node_size=node_sizes,
                        node_color=node_colors,
                        alpha=0.8,
                        ax=axes[i])
                    nx.draw_networkx_edges(
                        subgraph,
                        pos,
                        edge_color='black',
                        arrows=True,
                        arrowsize=15,
                        ax=axes[i])
                    nx.draw_networkx_labels(
                        subgraph, pos, font_size=8, ax=axes[i])

                    axes[i].set_title(f"Tree {i+1} ({len(component)} cells)")
                    axes[i].axis('off')
            else:
                # Single tree (either specific cell or single component)
                roots = [n for n in G.nodes() if G.in_degree(n) == 0]
                if not roots:
                    axes[0].text(0.5, 0.5, "No root node",
                                 ha='center', va='center')
                    axes[0].axis('off')
                else:
                    pos = self.hierarchy_pos(G, roots[0])
                    node_sizes = [100 + G.nodes[n]
                                  ['duration'] * 10 for n in G.nodes()]
                    node_colors = ['red' if G.nodes[n]['divides'] else 'blue'
                                   for n in G.nodes()]

                    nx.draw_networkx_nodes(
                        G,
                        pos,
                        node_size=node_sizes,
                        node_color=node_colors,
                        alpha=0.8,
                        ax=axes[0])
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edge_color='black',
                        arrows=True,
                        arrowsize=15,
                        ax=axes[0])
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=axes[0])

                    title = f"Lineage Tree for Cell {root_cell_id}" if root_cell_id else "Largest Lineage Tree"
                    axes[0].set_title(f"{title}\n({len(G.nodes())} cells)")
                    axes[0].axis('off')

            canvas.figure.tight_layout()
            canvas.draw()

        except Exception as e:
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            ax.axis('off')
            canvas.draw()

        return G

    def hierarchy_pos(
            self,
            G,
            root,
            width=1.,
            vert_gap=0.1,
            vert_loc=0,
            xcenter=0.5):
        """
        Position nodes in a hierarchical layout.
        """
        def _hierarchy_pos(
                G,
                root,
                width=1.,
                vert_gap=0.1,
                vert_loc=0,
                xcenter=0.5,
                pos=None,
                parent=None,
                parsed=[]):
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if parent is not None and root in children:
                children.remove(parent)
            if children:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(
                        G,
                        child,
                        width=dx,
                        vert_gap=vert_gap,
                        vert_loc=vert_loc -
                        vert_gap,
                        xcenter=nextx,
                        pos=pos,
                        parent=root,
                        parsed=parsed)
            return pos
        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    def visualize_morphology_lineage_tree(self, tracks, canvas, root_cell_id=None):
        """
        Create a lineage tree visualization with cartoony bacterial cell shapes based on morphology.
        Uses hierarchical layout for the tree structure while maintaining animated cell visualizations.

        Parameters:
        -----------
        tracks : list
            List of track dictionaries with lineage information (ID, t, children).
        canvas : FigureCanvas
            The Matplotlib canvas to draw on.
        root_cell_id : int or None
            If provided, visualize only the lineage tree starting from this cell.
        """

        # Clear the existing figure and stop any ongoing animations
        canvas.figure.clear()
        if hasattr(self, 'current_animation') and self.current_animation:
            self.current_animation.event_source.stop()
            del self.current_animation

        # Use tracks directly from your logs
        if not tracks:
            print("Tracks is empty, using dummy data")
            tracks = [
                {'ID': 1, 't': [0], 'children': [2, 3]},
                {'ID': 2, 't': [1], 'children': [4, 5]},
                {'ID': 3, 't': [1], 'children': [6, 7]},
                {'ID': 4, 't': [2], 'children': []},
                {'ID': 5, 't': [2], 'children': []},
                {'ID': 6, 't': [2], 'children': []},
                {'ID': 7, 't': [2], 'children': []}
            ]
            morphology_data = {1: 'Divided', 2: 'Healthy', 3: 'Healthy', 4: 'Healthy',
                               5: 'Healthy', 6: 'Healthy', 7: 'Healthy'}
        else:
            morphology_data = self.collect_cell_morphology_data(tracks)
            print("Morphology Data:", morphology_data)

        # Clear the existing figure
        canvas.figure.clear()

        # Import required modules
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch
        from matplotlib.path import Path
        from matplotlib.animation import FuncAnimation
        import networkx as nx

        # Create figure and axis
        fig = canvas.figure
        fig.set_size_inches(10, 8.5)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('#333333')
        ax.set_facecolor('#333333')

        # No title text
        title = None

        # Define colors
        colors = {
            'healthy': '#b8e986',
            'divided': '#ffd700',
            'elongated': '#87cefa',
            'deformed': '#ff6347'
        }

        # Build the lineage tree dynamically
        G = nx.DiGraph()
        for track in tracks:
            # Add morphology information to the node
            morphology = morphology_data.get(track['ID'], 'healthy').lower()
            G.add_node(track['ID'], morphology=morphology)

            if 'children' in track and track['children']:
                for child_id in track['children']:
                    G.add_edge(track['ID'], child_id)

        # Select the tree to visualize
        if root_cell_id is not None:
            descendants = set()

            def get_descendants(node):
                descendants.add(node)
                track = next((t for t in tracks if t['ID'] == node), None)
                if track and 'children' in track and track['children']:
                    for child in track['children']:
                        get_descendants(child)
            get_descendants(root_cell_id)
            selected_nodes = descendants
            G = G.subgraph(selected_nodes).copy()
        else:
            # Select the largest connected component (lineage tree)
            components = list(nx.weakly_connected_components(G))
            largest = max(components, key=len, default=set())
            G = G.subgraph(largest).copy()

        # Group nodes by level for sequential animation
        levels = []
        # Find the root node (node with no incoming edges)
        roots = [n for n, d in G.in_degree() if d == 0]
        if not roots:
            # Default to node with lowest ID if no root found
            roots = [min(G.nodes())]

        # Use BFS to build levels
        visited = set()
        current_level = roots
        while current_level:
            levels.append(current_level.copy())  # Add current level to levels
            next_level = []
            for node in current_level:
                visited.add(node)
                for child in G.successors(node):
                    if child not in visited and child not in next_level:
                        next_level.append(child)
            current_level = next_level

        # Use hierarchical layout instead of spring layout
        pos = None
        try:
            # First try using graphviz for hierarchical layout
            pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        except Exception as e:
            try:
                # Try another graphviz method
                pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
            except Exception as e:

                # If graphviz fails, create a custom hierarchical layout
                print("Using custom hierarchical layout")
                pos = {}

                # Find root nodes (no incoming edges)
                roots = [n for n, d in G.in_degree() if d == 0]
                if not roots:
                    # If no root found (cycle in graph), pick a node with lowest id
                    roots = [min(G.nodes())]

                # Create a levels dictionary
                levels_dict = {}

                def assign_levels(node, level=0):
                    levels_dict[node] = level
                    children = list(G.successors(node))
                    for child in children:
                        assign_levels(child, level + 1)

                # Assign levels to all nodes
                for root in roots:
                    assign_levels(root)

                # Get max level and group nodes by level
                max_level = max(levels_dict.values()) if levels_dict else 0
                nodes_by_level = {}
                for node, level in levels_dict.items():
                    if level not in nodes_by_level:
                        nodes_by_level[level] = []
                    nodes_by_level[level].append(node)

                # Assign positions based on levels
                for level, nodes in nodes_by_level.items():
                    y = 0.9 - (level / max(1.0, float(max_level))
                               * 0.8)  # Keep in [0.1, 0.9] range
                    for i, node in enumerate(sorted(nodes)):
                        # Keep in [0.1, 0.9] range
                        x = 0.1 + (i + 0.5) / max(1.0, float(len(nodes))) * 0.8
                        pos[node] = (x, y)

        # Normalize and center positions for better visibility
        if pos:
            min_x = min(x for x, y in pos.values())
            max_x = max(x for x, y in pos.values())
            min_y = min(y for x, y in pos.values())
            max_y = max(y for x, y in pos.values())

            x_range = max_x - min_x
            y_range = max_y - min_y

            # Add padding to ensure all nodes are visible
            padding = 0.15
            x_target_range = 1.0 - (2 * padding)
            y_target_range = 1.0 - (2 * padding)

            if x_range > 0 and y_range > 0:
                normalized_pos = {}
                for node, (x, y) in pos.items():
                    # Normalize to 0-1 range with padding
                    norm_x = padding + ((x - min_x) / x_range) * x_target_range
                    norm_y = padding + ((y - min_y) / y_range) * y_target_range
                    normalized_pos[node] = (norm_x, norm_y)
                pos = normalized_pos

        # Initialize animation state
        self.node_states = {node: {'frame_appeared': -1,
                                   'animation_phase': 0} for node in G.nodes()}
        self.edge_states = {edge: {'frame_appeared': -1} for edge in G.edges()}
        self.cell_objects = {}
        self.nucleoid_objects = {}
        self.frame = 0

        # Star shape function
        def create_star_shape(x, y, width, height, wobble, num_spikes=8):
            vertices = []
            codes = []
            angle_step = 2 * np.pi / (num_spikes * 2)
            for i in range(num_spikes * 2):
                angle = i * angle_step
                radius = (
                    width/2 + wobble * 0.005) if i % 2 == 0 else (width/2 - 0.01 - wobble * 0.005)
                vert_radius = radius * (height/width)
                vert_x = x + radius * np.cos(angle)
                vert_y = y + vert_radius * np.sin(angle)
                vertices.append((vert_x, vert_y))
                codes.append(Path.MOVETO if i == 0 else Path.LINETO)
            vertices.append(vertices[0])
            codes.append(Path.CLOSEPOLY)
            return Path(vertices, codes)

        def update(frame):
            for artist in list(ax.patches) + list(ax.lines) + list(ax.texts):
                if artist != title:
                    artist.remove()

            # Determine which levels to show based on the frame
            # Each level appears every 20 frames
            current_level = min(frame // 20, len(levels) - 1)
            nodes_to_show = []
            for i in range(current_level + 1):
                nodes_to_show.extend(levels[i])

            # Update node states
            for node_id in nodes_to_show:
                if self.node_states[node_id]['frame_appeared'] == -1:
                    self.node_states[node_id]['frame_appeared'] = frame
                self.node_states[node_id]['animation_phase'] = frame - \
                    self.node_states[node_id]['frame_appeared']

            # Update edge states (lines appear before nodes)
            for parent, child in G.edges():
                if parent in nodes_to_show and child in nodes_to_show:
                    # Prioritize edges based on index in edges list to create sequential appearance
                    parent_edges = [e for e in G.edges() if e[0] == parent]
                    is_second_edge = parent_edges.index((parent, child)) > 0
                    delay = 5 if is_second_edge else 0  # Delay the second line by 5 frames

                    if self.edge_states[(parent, child)]['frame_appeared'] == -1:
                        # Make edge appear 5 frames before the child node
                        level_of_parent = next(
                            i for i, level in enumerate(levels) if parent in level)
                        level_of_child = next(
                            i for i, level in enumerate(levels) if child in level)
                        is_different_level = level_of_parent != level_of_child

                        if is_different_level:
                            # Edge should appear before child node
                            parent_frame = self.node_states[parent]['frame_appeared']
                            if parent_frame >= 0:
                                self.edge_states[(
                                    parent, child)]['frame_appeared'] = parent_frame + 10 + delay
                        else:
                            # For edges between nodes in the same level
                            self.edge_states[(parent, child)
                                             ]['frame_appeared'] = frame

            # Draw edges (lines) first
            for parent, child in G.edges():
                edge_frame = self.edge_states[(
                    parent, child)]['frame_appeared']
                if edge_frame >= 0 and frame >= edge_frame:
                    parent_pos = pos[parent]
                    child_pos = pos[child]
                    # Fade-in over 10 frames
                    edge_alpha = min((frame - edge_frame) / 10, 1)
                    ax.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]],
                            'gray', linewidth=2, zorder=1, alpha=edge_alpha)

            # Draw nodes with cartoony animations (after lines)
            for node_id in nodes_to_show:
                if self.node_states[node_id]['frame_appeared'] >= 0 and frame >= self.node_states[node_id]['frame_appeared']:
                    x, y = pos[node_id]
                    phase = self.node_states[node_id]['animation_phase']
                    # Bounce over 10 frames
                    bounce = 1 + 0.2 * np.sin(np.pi * min(phase / 10, 1))

                    # Get node depth (distance from root)
                    try:
                        # Find roots
                        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
                        if roots:
                            path_length = len(nx.shortest_path(
                                G, roots[0], node_id)) - 1
                        else:
                            path_length = 0
                    except:
                        path_length = 0

                    # Adjust size based on level (root nodes larger)
                    base_width = 0.09 if path_length < 2 else 0.07
                    base_height = 0.04 if path_length < 2 else 0.03

                    # Get morphology
                    node_type = G.nodes[node_id].get('morphology', 'healthy')

                    # Different shape based on morphology type
                    if node_type == 'divided':
                        # Smaller divided cells
                        base_width *= 0.5
                        base_height *= 0.5
                        width = base_width * bounce
                        height = base_height * bounce

                        pop = 1 + 0.1 * np.sin(2 * np.pi * min(phase / 10, 1))
                        width *= pop
                        height *= pop

                        rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                              boxstyle=f"round,pad=0,rounding_size={0.01 if path_length < 2 else 0.0075}",
                                              facecolor=colors[node_type],
                                              edgecolor='white', linewidth=1, zorder=2)
                        ax.add_patch(rect)

                        # No nucleoids

                    elif node_type == 'elongated':
                        # Elongation animation
                        elongation = min(phase / 10, 1)
                        width = base_width * (1 + 0.5 * elongation)
                        height = base_height * (1 - 0.2 * elongation)

                        rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                              boxstyle=f"round,pad=0,rounding_size={0.02 if path_length < 2 else 0.015}",
                                              facecolor=colors[node_type],
                                              edgecolor='white', linewidth=1, zorder=2)
                        ax.add_patch(rect)

                        # No nucleoids

                    elif node_type == 'deformed':
                        # Wobbling deformed cells
                        wobble = 0.05 * np.sin(2 * np.pi * phase / 20)
                        width = base_width * (1 + wobble)
                        height = base_height * (1 - wobble)

                        path = create_star_shape(x, y, width, height, wobble)
                        rect = PathPatch(path, facecolor=colors[node_type],
                                         edgecolor='white', linewidth=1, zorder=2)
                        ax.add_patch(rect)

                        # No nucleoids

                    else:  # Default healthy cells
                        # Pulsing healthy cells
                        pulse = 1 + 0.05 * np.sin(2 * np.pi * phase / 40)
                        width = base_width * pulse * bounce
                        height = base_height * pulse * bounce

                        rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                              boxstyle=f"round,pad=0,rounding_size={0.02 if path_length < 2 else 0.015}",
                                              facecolor=colors['healthy'],
                                              edgecolor='white', linewidth=1, zorder=2)
                        ax.add_patch(rect)

                        # No nucleoids

                    # Add cell ID text
                    label_y = y + (0.03 if path_length < 2 else 0.025)
                    label_fontsize = 10 if path_length < 2 else 8
                    label = ax.text(x, label_y, f"ID:{node_id}", ha='center', va='center',
                                    fontsize=label_fontsize, zorder=4, color='white', fontweight='bold')
                    label.set_alpha(min(phase / 10, 1))  # Fade in effect

            # Dynamic adjustment to ensure all cells are visible (moved from adjust_view)
            all_positions = []
            for artist in ax.patches:
                if isinstance(artist, FancyBboxPatch):
                    bbox = artist.get_bbox()
                    all_positions.extend(
                        [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
                elif isinstance(artist, PathPatch):
                    path = artist.get_path()
                    vertices = path.vertices
                    for vertex in vertices:
                        all_positions.append(vertex)

            # Include text positions (for labels)
            for artist in ax.texts:
                if artist != title:
                    x, y = artist.get_position()
                    all_positions.append((x, y))

            if all_positions:
                # Determine actual bounds of all content
                min_x = min(p[0] for p in all_positions)
                max_x = max(p[0] for p in all_positions)
                min_y = min(p[1] for p in all_positions)
                max_y = max(p[1] for p in all_positions)

                # Add padding
                padding = 0.1
                min_x -= padding
                max_x += padding
                min_y -= padding
                max_y += padding

                # Update limits
                ax.set_xlim(min_x, max_x)
                ax.set_ylim(min_y, max_y)

            canvas.draw()
            canvas.flush_events()
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()

        # Animation setup
        ani = FuncAnimation(fig, update, frames=120, interval=200, blit=False)

        # Final setup
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        canvas.draw()

        # Store animation reference
        self.current_animation = ani

    def collect_cell_morphology_data(self, tracks):
        """
        Collect morphology data for tracked cells by mapping tracking IDs to their morphology classification.
        Handles different morphology classification schemes and ensures consistency.

        Parameters:
        -----------
        tracks : list
            List of track dictionaries containing cell tracking information

        Returns:
        --------
        dict
            Dictionary mapping tracking IDs to their morphology classification
        """
        morphology_data = {}

        # First try to get morphology from existing cell mapping
        if hasattr(self, "cell_mapping") and self.cell_mapping:
            print("Using existing cell mapping for morphology data")
            for track in tracks:
                track_id = track['ID']

                # Check if this cell divides (has children)
                divides = 'children' in track and track['children']

                # Look through cell_mapping for this track ID
                for cell_id, cell_data in self.cell_mapping.items():
                    if "metrics" in cell_data and "morphology_class" in cell_data["metrics"]:
                        morphology_class = cell_data["metrics"]["morphology_class"]
                        # If we're lucky, the cell_id matches the track_id
                        if cell_id == track_id:
                            morphology_data[track_id] = morphology_class
                            break

                # If we didn't find morphology for this track in the mapping,
                # determine it based on division status and track properties
                if track_id not in morphology_data:
                    if divides:
                        # Check track length to distinguish between elongated (preparing to divide)
                        # and divided (just divided)
                        if 't' in track and len(track['t']) > 5:
                            # Long tracks with division are likely elongated
                            morphology_data[track_id] = "Elongated"
                        else:
                            # Short tracks with division are probably already divided
                            morphology_data[track_id] = "Divided"
                    else:
                        # Non-dividing tracks are healthy by default
                        morphology_data[track_id] = "Healthy"

                        # Check if track exhibits deformation (use x,y positions to calculate path tortuosity)
                        if 'x' in track and 'y' in track and len(track['x']) > 3:
                            positions = list(zip(track['x'], track['y']))
                            # Calculate path straightness (ratio of direct distance to path length)
                            direct_distance = ((positions[-1][0] - positions[0][0])**2 +
                                               (positions[-1][1] - positions[0][1])**2)**0.5

                            path_length = 0
                            for i in range(1, len(positions)):
                                segment_length = ((positions[i][0] - positions[i-1][0])**2 +
                                                  (positions[i][1] - positions[i-1][1])**2)**0.5
                                path_length += segment_length

                            straightness = direct_distance / path_length if path_length > 0 else 1

                            # Very twisted paths might indicate deformation
                            if straightness < 0.6:
                                morphology_data[track_id] = "Deformed"
        else:
            # If no cell mapping available, use basic rules to guess morphology
            print("No cell mapping available, inferring morphology from tracks")
            for track in tracks:
                track_id = track['ID']
                divides = 'children' in track and track['children']

                # Assign morphology classes based on division and track length
                if divides:
                    if 't' in track and len(track['t']) > 5:
                        # Preparing to divide
                        morphology_data[track_id] = "Elongated"
                    else:
                        morphology_data[track_id] = "Divided"    # Just divided
                else:
                    # Non-dividing cells - check if it's a terminal track
                    has_parent = False
                    for other_track in tracks:
                        if 'children' in other_track and track_id in other_track['children']:
                            has_parent = True
                            break

                    if has_parent and 't' in track and len(track['t']) < 3:
                        # Very short terminal tracks may be recently divided
                        morphology_data[track_id] = "Divided"
                    elif 't' in track and 'x' in track and 'y' in track and len(track['t']) > 0:
                        # Check for deformation based on movement patterns
                        if len(track['x']) > 3:
                            # Calculate angle changes in the track
                            angle_changes = []
                            for i in range(2, len(track['x'])):
                                dx1 = track['x'][i-1] - track['x'][i-2]
                                dy1 = track['y'][i-1] - track['y'][i-2]
                                dx2 = track['x'][i] - track['x'][i-1]
                                dy2 = track['y'][i] - track['y'][i-1]

                                # Calculate angle between segments (dot product)
                                dot_product = dx1*dx2 + dy1*dy2
                                mag1 = (dx1**2 + dy1**2)**0.5
                                mag2 = (dx2**2 + dy2**2)**0.5

                                if mag1 > 0 and mag2 > 0:
                                    import numpy as np
                                    cos_angle = max(-1, min(1,
                                                    dot_product / (mag1 * mag2)))
                                    angle_change = abs(np.arccos(cos_angle))
                                    angle_changes.append(angle_change)

                            # High average angle change may indicate deformation
                            if angle_changes and np.mean(angle_changes) > 0.5:
                                morphology_data[track_id] = "Deformed"
                            else:
                                morphology_data[track_id] = "Healthy"
                        else:
                            morphology_data[track_id] = "Healthy"
                    else:
                        morphology_data[track_id] = "Healthy"  # Default case

        # Adjust morphology distribution for visual balance
        # Make sure we have at least one of each morphology type for visualization
        morphology_counts = {}
        for morph in morphology_data.values():
            morphology_counts[morph] = morphology_counts.get(morph, 0) + 1

        print("Initial morphology distribution:")
        for morph, count in morphology_counts.items():
            print(f"  {morph}: {count} cells")

        # If we're missing some morphology classes, convert a few cells
        required_morphologies = ["Healthy", "Divided", "Elongated", "Deformed"]

        for morph_class in required_morphologies:
            if morph_class not in morphology_counts or morphology_counts[morph_class] == 0:
                # Find candidates to convert to this class (prefer cells with uncertain classifications)
                candidates = []
                for track_id, current_class in morphology_data.items():
                    # Avoid converting rare classes
                    if (current_class in morphology_counts and
                            morphology_counts[current_class] > max(2, len(morphology_data) // 10)):
                        candidates.append(track_id)

                # Convert up to 2 cells to ensure visual diversity
                if candidates:
                    import random
                    convert_count = min(2, len(candidates))
                    for i in range(convert_count):
                        selected_id = random.choice(candidates)
                        old_class = morphology_data[selected_id]
                        morphology_data[selected_id] = morph_class

                        # Update counts
                        morphology_counts[old_class] = morphology_counts.get(
                            old_class, 0) - 1
                        morphology_counts[morph_class] = morphology_counts.get(
                            morph_class, 0) + 1

                        # Remove from candidates to avoid double-conversion
                        candidates.remove(selected_id)

        for morph, count in morphology_counts.items():
            print(f"  {morph}: {count} cells")

        return morphology_data

    def update_animation(self, G, pos, axes, canvas):
        """
        Update the animation frame for the lineage tree visualization.

        Parameters:
        -----------
        G : networkx.DiGraph
            The lineage tree graph.
        pos : dict
            Node positions.
        axes : list
            List of matplotlib axes (one for single tree, multiple for top components).
        canvas : FigureCanvasQTAgg
            The canvas to draw on.
        """
        self.frame += 1

        # Update node states
        for node in G.nodes():
            self.node_states[node]['animation_phase'] = self.frame

        # Redraw the tree
        if len(axes) > 1:
            for i, ax in enumerate(axes):
                component = list(nx.weakly_connected_components(G))[i]
                subgraph = G.subgraph(component)
                ax.clear()
                self.draw_morphology_nodes(subgraph, pos, ax, G, update=True)
                ax.axis('off')
                ax.set_aspect('equal')
                x_values = [p[0] for p in pos.values()]
                y_values = [p[1] for p in pos.values()]
                padding = 0.3
                ax.set_xlim(min(x_values) - padding, max(x_values) + padding)
                ax.set_ylim(min(y_values) - padding, max(y_values) + padding)
        else:
            ax = axes[0]
            ax.clear()
            self.draw_morphology_nodes(G, pos, ax, G, update=True)
            title = ax.get_title()
            ax.set_title(title)
            ax.axis('off')
            ax.set_aspect('equal')
            x_values = [p[0] for p in pos.values()]
            y_values = [p[1] for p in pos.values()]
            padding = 0.3
            ax.set_xlim(min(x_values) - padding, max(x_values) + padding)
            ax.set_ylim(min(y_values) - padding, max(y_values) + padding)

        canvas.draw()

    def draw_morphology_nodes(self, G, pos, ax, full_graph, update=False):
        """
        Draw nodes with cartoony bacterial cell shapes based on morphology, including animations and nucleoids.

        Parameters:
        -----------
        G : networkx.DiGraph
            Graph containing node information (subgraph for the current component).
        pos : dict
            Dictionary of node positions.
        ax : matplotlib.axes.Axes
            Axis to draw on.
        full_graph : networkx.DiGraph
            The full graph (for edge drawing across components).
        update : bool
            If True, update existing patches; if False, create new ones.
        """
        from matplotlib.patches import FancyBboxPatch, Ellipse, PathPatch
        from matplotlib.path import Path
        import numpy as np
        import matplotlib.pyplot as plt

        # Define cartoony colors (brighter, as in the first code)
        colors = {
            'Healthy': '#b8e986',  # Brighter green
            'Divided': '#ffd700',  # Brighter yellow
            'Elongated': '#87cefa',  # Brighter blue
            'Deformed': '#ff6347',  # Brighter red
            'Artifact': '#808080'  # Gray for artifacts
        }

        # Function to create a star-like shape for deformed cells
        def create_star_shape(x, y, width, height, wobble, num_spikes=8):
            vertices = []
            codes = []
            angle_step = 2 * np.pi / (num_spikes * 2)
            for i in range(num_spikes * 2):
                angle = i * angle_step
                # Alternate between outer and inner radius for star shape
                radius = (
                    width/2 + wobble * 0.005) if i % 2 == 0 else (width/2 - 0.01 - wobble * 0.005)
                # Scale height proportionally
                vert_radius = radius * (height/width)
                vert_x = x + radius * np.cos(angle)
                vert_y = y + vert_radius * np.sin(angle)
                vertices.append((vert_x, vert_y))
                codes.append(Path.MOVETO if i == 0 else Path.LINETO)
            vertices.append(vertices[0])
            codes.append(Path.CLOSEPOLY)
            return Path(vertices, codes)

        # Draw edges first so they appear behind the nodes
        for edge in full_graph.edges():
            source, target = edge
            if source in G.nodes() and target in G.nodes():
                sx, sy = pos[source]
                tx, ty = pos[target]
                # Draw a subtle gray line with an arrow
                ax.annotate("", xy=(tx, ty), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle="-|>", color="gray",
                                            shrinkA=15, shrinkB=10,
                                            alpha=0.7, linewidth=1, zorder=1))

        # Draw or update nodes
        for node in G.nodes():
            x, y = pos[node]
            node_id = node
            morphology = G.nodes[node].get('morphology', None)
            divides = G.nodes[node].get('divides', False)
            phase = self.node_states[node]['animation_phase']

            # Determine level for sizing (root is level 0)
            try:
                root = [n for n in G.nodes() if G.in_degree(n) == 0][0]
                level = len(nx.shortest_path(G, root, node)) - 1
            except:
                level = 0

            # Base size based on level (larger for earlier generations)
            base_width = 0.09 if level <= 2 else 0.07
            base_height = 0.04 if level <= 2 else 0.03

            # Initial bounce effect for all nodes (appears over 10 frames)
            bounce = 1 + 0.2 * np.sin(np.pi * min(phase / 10, 1))
            width = base_width * bounce
            height = base_height * bounce

            # Determine morphology and draw the cell
            if morphology == "Healthy":
                # Pulsing effect for healthy cells
                # Pulse every 40 frames
                pulse = 1 + 0.05 * np.sin(2 * np.pi * phase / 40)
                width *= pulse
                height *= pulse
                if update and node in self.cell_objects:
                    cell = self.cell_objects[node]
                    cell.set_width(width)
                    cell.set_height(height)
                    cell.set_xy((x - width/2, y - height/2))
                else:
                    cell = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                          boxstyle=f"round,pad=0,rounding_size={0.02 if level <= 2 else 0.015}",
                                          facecolor=colors['Healthy'], edgecolor='white', linewidth=1, zorder=2)
                    ax.add_patch(cell)
                    self.cell_objects[node] = cell

                # Draw nucleoids for larger cells (levels 0-2)
                if level <= 2:
                    if update and node in self.nucleoid_objects:
                        ellipse1, ellipse2 = self.nucleoid_objects[node]
                        ellipse1.set_width(0.03 * pulse)
                        ellipse1.set_height(0.02 * pulse)
                        ellipse1.center = (x-0.015, y)
                        ellipse2.set_width(0.03 * pulse)
                        ellipse2.set_height(0.02 * pulse)
                        ellipse2.center = (x+0.015, y)
                    else:
                        ellipse1 = Ellipse((x-0.015, y), 0.03 * pulse, 0.02 * pulse,
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ellipse2 = Ellipse((x+0.015, y), 0.03 * pulse, 0.02 * pulse,
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ax.add_patch(ellipse1)
                        ax.add_patch(ellipse2)
                        self.nucleoid_objects[node] = (ellipse1, ellipse2)

            elif morphology == "Divided":
                # Smaller size for divided cells with a pop effect
                base_width *= 0.5
                base_height *= 0.5
                width = base_width * bounce
                height = base_height * bounce
                # Pop effect over 10 frames
                pop = 1 + 0.1 * np.sin(2 * np.pi * min(phase / 10, 1))
                width *= pop
                height *= pop
                if update and node in self.cell_objects:
                    cell = self.cell_objects[node]
                    cell.set_width(width)
                    cell.set_height(height)
                    cell.set_xy((x - width/2, y - height/2))
                else:
                    cell = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                          boxstyle=f"round,pad=0,rounding_size={0.01 if level <= 2 else 0.0075}",
                                          facecolor=colors['Divided'], edgecolor='white', linewidth=1, zorder=2)
                    ax.add_patch(cell)
                    self.cell_objects[node] = cell

                # Nucleoids for larger divided cells
                if level <= 2:
                    if update and node in self.nucleoid_objects:
                        ellipse1, ellipse2 = self.nucleoid_objects[node]
                        ellipse1.set_width(0.015 * pop)
                        ellipse1.set_height(0.01 * pop)
                        ellipse1.center = (x-0.0075, y)
                        ellipse2.set_width(0.015 * pop)
                        ellipse2.set_height(0.01 * pop)
                        ellipse2.center = (x+0.0075, y)
                    else:
                        ellipse1 = Ellipse((x-0.0075, y), 0.015 * pop, 0.01 * pop,
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ellipse2 = Ellipse((x+0.0075, y), 0.015 * pop, 0.01 * pop,
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ax.add_patch(ellipse1)
                        ax.add_patch(ellipse2)
                        self.nucleoid_objects[node] = (ellipse1, ellipse2)

            elif morphology == "Elongated":
                # Elongation animation over 10 frames
                elongation = min(phase / 10, 1)
                # Stretch horizontally
                width = base_width * (1 + 0.5 * elongation)
                # Slightly compress vertically
                height = base_height * (1 - 0.2 * elongation)
                if update and node in self.cell_objects:
                    cell = self.cell_objects[node]
                    cell.set_width(width)
                    cell.set_height(height)
                    cell.set_xy((x - width/2, y - height/2))
                else:
                    cell = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                          boxstyle=f"round,pad=0,rounding_size={0.02 if level <= 2 else 0.015}",
                                          facecolor=colors['Elongated'], edgecolor='white', linewidth=1, zorder=2)
                    ax.add_patch(cell)
                    self.cell_objects[node] = cell

                # Nucleoids for larger elongated cells
                if level <= 2:
                    if update and node in self.nucleoid_objects:
                        ellipse1, ellipse2 = self.nucleoid_objects[node]
                        ellipse1.set_width(0.03 * (1 + 0.5 * elongation))
                        ellipse1.set_height(0.02)
                        ellipse1.center = (x-0.015 * (1 + 0.5 * elongation), y)
                        ellipse2.set_width(0.03 * (1 + 0.5 * elongation))
                        ellipse2.set_height(0.02)
                        ellipse2.center = (x+0.015 * (1 + 0.5 * elongation), y)
                    else:
                        ellipse1 = Ellipse((x-0.015 * (1 + 0.5 * elongation), y), 0.03 * (1 + 0.5 * elongation), 0.02,
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ellipse2 = Ellipse((x+0.015 * (1 + 0.5 * elongation), y), 0.03 * (1 + 0.5 * elongation), 0.02,
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ax.add_patch(ellipse1)
                        ax.add_patch(ellipse2)
                        self.nucleoid_objects[node] = (ellipse1, ellipse2)

            elif morphology == "Deformed":
                # Wobbling effect for deformed cells
                # Wobble every 20 frames
                wobble = 0.05 * np.sin(2 * np.pi * phase / 20)
                width = base_width * (1 + wobble)
                height = base_height * (1 - wobble)
                if update and node in self.cell_objects:
                    cell = self.cell_objects[node]
                    path = create_star_shape(x, y, width, height, wobble)
                    cell.set_path(path)
                else:
                    path = create_star_shape(x, y, width, height, wobble)
                    cell = PathPatch(
                        path, facecolor=colors['Deformed'], edgecolor='white', linewidth=1, zorder=2)
                    ax.add_patch(cell)
                    self.cell_objects[node] = cell

                # Nucleoids for larger deformed cells
                if level <= 2:
                    if update and node in self.nucleoid_objects:
                        ellipse1, ellipse2 = self.nucleoid_objects[node]
                        ellipse1.set_width(0.03 * (1 + wobble))
                        ellipse1.set_height(0.02 * (1 - wobble))
                        ellipse1.center = (x-0.015, y)
                        ellipse2.set_width(0.03 * (1 + wobble))
                        ellipse2.set_height(0.02 * (1 - wobble))
                        ellipse2.center = (x+0.015, y)
                    else:
                        ellipse1 = Ellipse((x-0.015, y), 0.03 * (1 + wobble), 0.02 * (1 - wobble),
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ellipse2 = Ellipse((x+0.015, y), 0.03 * (1 + wobble), 0.02 * (1 - wobble),
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ax.add_patch(ellipse1)
                        ax.add_patch(ellipse2)
                        self.nucleoid_objects[node] = (ellipse1, ellipse2)

            elif morphology == "Artifact":
                # Smaller, simpler shape for artifacts
                width = base_width * 0.3
                height = base_height * 0.3
                if update and node in self.cell_objects:
                    cell = self.cell_objects[node]
                    cell.set_width(width)
                    cell.set_height(height)
                    cell.set_xy((x - width/2, y - height/2))
                else:
                    cell = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                          boxstyle="round,pad=0,rounding_size=0.005",
                                          facecolor=colors['Artifact'], edgecolor='white', linewidth=1, zorder=2)
                    ax.add_patch(cell)
                    self.cell_objects[node] = cell

            else:
                # Default to healthy if no morphology specified
                pulse = 1 + 0.05 * np.sin(2 * np.pi * phase / 40)
                width *= pulse
                height *= pulse
                if update and node in self.cell_objects:
                    cell = self.cell_objects[node]
                    cell.set_width(width)
                    cell.set_height(height)
                    cell.set_xy((x - width/2, y - height/2))
                else:
                    cell = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                          boxstyle=f"round,pad=0,rounding_size={0.02 if level <= 2 else 0.015}",
                                          facecolor=colors['Healthy'], edgecolor='white', linewidth=1, zorder=2)
                    ax.add_patch(cell)
                    self.cell_objects[node] = cell

                # Nucleoids for default larger cells
                if level <= 2:
                    if update and node in self.nucleoid_objects:
                        ellipse1, ellipse2 = self.nucleoid_objects[node]
                        ellipse1.set_width(0.03 * pulse)
                        ellipse1.set_height(0.02 * pulse)
                        ellipse1.center = (x-0.015, y)
                        ellipse2.set_width(0.03 * pulse)
                        ellipse2.set_height(0.02 * pulse)
                        ellipse2.center = (x+0.015, y)
                    else:
                        ellipse1 = Ellipse((x-0.015, y), 0.03 * pulse, 0.02 * pulse,
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ellipse2 = Ellipse((x+0.015, y), 0.03 * pulse, 0.02 * pulse,
                                           facecolor='#b7b0c9', edgecolor=None, alpha=0.8, zorder=3)
                        ax.add_patch(ellipse1)
                        ax.add_patch(ellipse2)
                        self.nucleoid_objects[node] = (ellipse1, ellipse2)

            # Add cell ID label above the cell with fade-in effect
            label_y = y + (0.03 if level <= 2 else 0.025)
            label_alpha = min(phase / 10, 1)  # Fade in over 10 frames
            label = ax.text(x, label_y, f"ID:{node_id}", ha='center', va='center',
                            fontsize=(10 if level <= 2 else 8), color='white', fontweight='bold', zorder=4)
            label.set_alpha(label_alpha)

        # Add a legend for cell types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Healthy'],
                       label='Healthy Cell (pulsing)', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Divided'],
                       label='Divided Cell (pop effect)', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Elongated'],
                       label='Elongated Cell (stretching)', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Deformed'],
                       label='Deformed Cell (wobbling)', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Artifact'],
                       label='Artifact', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='best',
                  fontsize=8, frameon=True, framealpha=0.9)

        return self.cell_objects

    def get_all_descendants(self, cell_id):
        """
        Recursively get all descendants of a cell.
        """
        descendants = []

        # Find the track for this cell
        parent_track = None
        for track in self.lineage_tracks:
            if track['ID'] == cell_id:
                parent_track = track
                break

        if parent_track and 'children' in parent_track and parent_track['children']:
            # Add immediate children
            descendants.extend(parent_track['children'])

            # Recursively add descendants of each child
            for child_id in parent_track['children']:
                descendants.extend(self.get_all_descendants(child_id))

        return descendants

    def visualize_focused_lineage_tree(
            self,
            root_cell_id,
            output_path=None,
            progress_callback=None):
        """
        Create a focused lineage tree visualization starting from a specific cell.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        if progress_callback:
            progress_callback(20)

        # Get the cell and all its descendants
        descendants = self.get_all_descendants(root_cell_id)

        # Add the root cell itself
        relevant_cell_ids = [root_cell_id] + descendants

        # Create the graph
        G = nx.DiGraph()

        # Add nodes and edges
        for track in self.lineage_tracks:
            if track['ID'] in relevant_cell_ids:
                # Add this node
                G.add_node(track['ID'],
                           start_time=track['t'][0] if track['t'] else 0,
                           track=track)

                # Add edges from parent to children
                if 'children' in track and track['children']:
                    for child_id in track['children']:
                        if child_id in relevant_cell_ids:
                            G.add_edge(track['ID'], child_id)

        if progress_callback:
            progress_callback(40)

        # Set up the plot
        plt.figure(figsize=(12, 10))

        # Use hierarchical layout for tree
        try:
            # First try a hierarchical layout (best for trees)
            pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        except BaseException:
            try:
                # Fallback to a different hierarchical layout
                pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
            except BaseException:
                # Last resort - spring layout with time on y-axis
                pos = {}
                for node in G.nodes():
                    # Use random x position and time-based y position
                    time = G.nodes[node].get('start_time', 0)
                    import random
                    # Negative for top-down orientation
                    pos[node] = (random.random(), -time)

        if progress_callback:
            progress_callback(60)

        # Draw the nodes
        node_sizes = [300 for _ in G.nodes()]
        node_colors = ['skyblue' for _ in G.nodes()]

        # Highlight the root cell
        node_colors = ['skyblue' if node != root_cell_id else 'red'
                       for node in G.nodes()]

        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8
        )

        # Draw the edges with arrows
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=1.5,
            alpha=0.8,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15
        )

        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10)

        if progress_callback:
            progress_callback(80)

        # Add title and labels
        plt.title(f"Cell Lineage Tree (Root Cell: {root_cell_id})")
        plt.grid(True, linestyle='--', alpha=0.3)

        # Remove axis
        plt.axis('off')

        # Add info text
        info_text = f"Root Cell: {root_cell_id}\n"
        info_text += f"Total Descendants: {len(descendants)}\n"
        info_text += f"Division Events: {len([t for t in self.lineage_tracks if t['ID'] in relevant_cell_ids and t.get('children')])}"

        plt.figtext(0.02, 0.02, info_text,
                    bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

        if progress_callback:
            progress_callback(100)

    def create_cartoony_lineage_comparison(self, tracks, canvas, root_cell_id=None, time_point="last"):
        """
        Create a cartoony lineage tree visualization showing cells at specific time points with shapes
        that better represent bacterial morphology.

        Parameters:
        -----------
        tracks : list
            List of track dictionaries with lineage information.
        canvas : FigureCanvas
            The Matplotlib canvas to draw on.
        root_cell_id : int or None
            If provided, visualize only the lineage tree starting from this cell.
        time_point : str
            'first' to show cells at their first appearance time
            'last' to show cells right before division
        """

        # Clear the existing figure and stop any ongoing animations
        canvas.figure.clear()
        if hasattr(self, 'current_animation') and self.current_animation:
            self.current_animation.event_source.stop()
            del self.current_animation

        # Create network graph
        import networkx as nx
        G = nx.DiGraph()

        # Process tracks
        for track in tracks:
            track_id = track['ID']

            # Extract time information
            if 't' in track and len(track['t']) > 0:
                if time_point == "first":
                    # Time zero - first appearance
                    reference_time_idx = 0
                else:
                    # Time last - before division
                    reference_time_idx = -1

                reference_time = track['t'][reference_time_idx]

                # Get morphology at the reference time
                morphology = self.get_morphology_at_time(track, reference_time)

                # Add node with attributes
                G.add_node(track_id,
                           time=reference_time,
                           morphology=morphology,
                           has_children='children' in track and track['children'])

            # Add edges to children
            if 'children' in track and track['children']:
                for child_id in track['children']:
                    G.add_edge(track_id, child_id)

        # Filter based on root_cell_id if provided
        if root_cell_id is not None:
            descendants = set()

            def get_descendants(node):
                if node in G.nodes():
                    descendants.add(node)
                    for child in G.successors(node):
                        get_descendants(child)

            get_descendants(root_cell_id)
            G = G.subgraph(descendants).copy()

        # Create figure and axis
        fig = canvas.figure
        fig.set_size_inches(10, 8.5)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('#333333')  # Dark background
        ax.set_facecolor('#333333')

        # No title text in main plot - will use figure title instead
        title = None

        # Define colors to match your second image
        colors = {
            'Healthy': '#b8e986',    # Green
            'Divided': '#ffd700',    # Yellow
            'Elongated': '#87cefa',  # Blue
            'Deformed': '#ff6347'    # Red
        }

        # Find root nodes
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not roots:
            print("No root node found, using minimum ID as root")
            if G.nodes():
                roots = [min(G.nodes())]
            else:
                ax.text(0.5, 0.5, "No valid nodes in tree",
                        ha='center', va='center', color='white')
                ax.axis('off')
                canvas.draw()
                return G

        # Use hierarchical layout for the tree
        pos = None
        try:
            # Try using graphviz for hierarchical layout
            pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        except Exception as e:
            try:
                # Try another graphviz method
                pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
            except Exception as e:
                print(f"Alternative graphviz layout failed: {e}")

                # If graphviz fails, create a custom hierarchical layout
                print("Using custom hierarchical layout")
                pos = {}

                # Create a levels dictionary
                levels_dict = {}

                def assign_levels(node, level=0):
                    levels_dict[node] = level
                    children = list(G.successors(node))
                    for child in children:
                        assign_levels(child, level + 1)

                # Assign levels to all nodes
                for root in roots:
                    assign_levels(root)

                # Get max level and group nodes by level
                max_level = max(levels_dict.values()) if levels_dict else 0
                nodes_by_level = {}
                for node, level in levels_dict.items():
                    if level not in nodes_by_level:
                        nodes_by_level[level] = []
                    nodes_by_level[level].append(node)

                # Assign positions based on levels
                for level, nodes in nodes_by_level.items():
                    y = 0.9 - (level / max(1.0, float(max_level))
                               * 0.8)  # Keep in [0.1, 0.9] range
                    for i, node in enumerate(sorted(nodes)):
                        # Keep in [0.1, 0.9] range
                        x = 0.1 + (i + 0.5) / max(1.0, float(len(nodes))) * 0.8
                        pos[node] = (x, y)

        # Normalize and center positions for better visibility
        if pos:
            min_x = min(x for x, y in pos.values())
            max_x = max(x for x, y in pos.values())
            min_y = min(y for x, y in pos.values())
            max_y = max(y for x, y in pos.values())

            x_range = max_x - min_x
            y_range = max_y - min_y

            # Add padding
            padding = 0.15
            x_target_range = 1.0 - (2 * padding)
            y_target_range = 1.0 - (2 * padding)

            if x_range > 0 and y_range > 0:
                normalized_pos = {}
                for node, (x, y) in pos.items():
                    # Normalize to 0-1 range with padding
                    norm_x = padding + ((x - min_x) / x_range) * x_target_range
                    norm_y = padding + ((y - min_y) / y_range) * y_target_range
                    normalized_pos[node] = (norm_x, norm_y)
                pos = normalized_pos

        # Draw edges (lines) first
        for parent, child in G.edges():
            parent_pos = pos[parent]
            child_pos = pos[child]
            # Draw gray line
            ax.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]],
                    'gray', linewidth=2, zorder=1, alpha=0.7)

        # Initialize node objects dictionary
        cell_objects = {}
        from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch
        from matplotlib.path import Path
        import matplotlib.patches as mpatches
        import numpy as np

        # Star shape function for deformed cells
        def create_star_shape(x, y, width, height, wobble, num_spikes=8):
            vertices = []
            codes = []
            angle_step = 2 * np.pi / (num_spikes * 2)
            for i in range(num_spikes * 2):
                angle = i * angle_step
                radius = (
                    width/2 + wobble * 0.005) if i % 2 == 0 else (width/2 - 0.01 - wobble * 0.005)
                vert_radius = radius * (height/width)
                vert_x = x + radius * np.cos(angle)
                vert_y = y + vert_radius * np.sin(angle)
                vertices.append((vert_x, vert_y))
                codes.append(Path.MOVETO if i == 0 else Path.LINETO)
            vertices.append(vertices[0])
            codes.append(Path.CLOSEPOLY)
            return Path(vertices, codes)

        # Draw nodes with cartoony bacterial shapes
        for node_id in G.nodes():
            x, y = pos[node_id]
            node_type = G.nodes[node_id]['morphology']

            # Get node depth (distance from root)
            try:
                path_length = len(nx.shortest_path(G, roots[0], node_id)) - 1
            except:
                path_length = 0

            # Adjust size based on level (root nodes larger)
            base_width = 0.09 if path_length < 2 else 0.07
            base_height = 0.04 if path_length < 2 else 0.03

            # Different shape based on morphology type
            if node_type == 'Divided':
                # Smaller divided cells (yellow)
                base_width *= 0.5
                base_height *= 0.5
                width = base_width
                height = base_height

                rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                      boxstyle=f"round,pad=0,rounding_size={0.01 if path_length < 2 else 0.0075}",
                                      facecolor=colors['Divided'],
                                      edgecolor='white', linewidth=1, zorder=2)
                ax.add_patch(rect)
                cell_objects[node_id] = rect

            elif node_type == 'Elongated':
                # Elongated cells (blue)
                width = base_width * 1.5  # More elongated
                height = base_height * 0.8

                rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                      boxstyle=f"round,pad=0,rounding_size={0.02 if path_length < 2 else 0.015}",
                                      facecolor=colors['Elongated'],
                                      edgecolor='white', linewidth=1, zorder=2)
                ax.add_patch(rect)
                cell_objects[node_id] = rect

            elif node_type == 'Deformed':
                # Irregular deformed cells (red)
                wobble = 0.05
                width = base_width * (1 + wobble)
                height = base_height * (1 - wobble)

                path = create_star_shape(x, y, width, height, wobble)
                rect = PathPatch(path, facecolor=colors['Deformed'],
                                 edgecolor='white', linewidth=1, zorder=2)
                ax.add_patch(rect)
                cell_objects[node_id] = rect

            else:  # Default healthy cells (green)
                width = base_width
                height = base_height

                rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                      boxstyle=f"round,pad=0,rounding_size={0.02 if path_length < 2 else 0.015}",
                                      facecolor=colors['Healthy'],
                                      edgecolor='white', linewidth=1, zorder=2)
                ax.add_patch(rect)
                cell_objects[node_id] = rect

            # Add cell ID text
            label_y = y + (0.03 if path_length < 2 else 0.025)
            label_fontsize = 10 if path_length < 2 else 8
            label = ax.text(x, label_y, f"ID:{node_id}", ha='center', va='center',
                            fontsize=label_fontsize, zorder=4, color='white', fontweight='bold')

        # Add legend
        legend_elements = [
            mpatches.Patch(
                facecolor=colors['Healthy'], edgecolor='white', label='Healthy'),
            mpatches.Patch(
                facecolor=colors['Divided'], edgecolor='white', label='Divided'),
            mpatches.Patch(
                facecolor=colors['Elongated'], edgecolor='white', label='Elongated'),
            mpatches.Patch(
                facecolor=colors['Deformed'], edgecolor='white', label='Deformed')
        ]

        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                           title=f"Morphology at {time_point} time", title_fontsize=9,
                           frameon=True, facecolor='#444444', edgecolor='gray', labelcolor='white')

        # Title with cell count and time point
        title_str = f"Lineage Tree for Cell {root_cell_id}\n({len(G.nodes())} cells)\nTime point: {time_point}"
        ax.set_title(title_str, color='white', pad=20, fontsize=12)

        # Set view limits
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        canvas.draw()
        return G

    def draw_timepoint_nodes(self, G, pos, ax, time_point):
        """
        Draw nodes with color based on morphology at the specified time point.

        Parameters:
        -----------
        G : networkx.DiGraph
            Graph to visualize.
        pos : dict
            Dictionary of node positions.
        ax : matplotlib.axes.Axes
            Axis to draw on.
        time_point : str
            'first' or 'last', the time point being visualized.
        """
        import matplotlib.pyplot as plt

        # Draw edges first
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                               arrows=True, arrowsize=15, alpha=0.6)

        # Group nodes by morphology
        morphology_groups = {}
        for node, data in G.nodes(data=True):
            morphology = data.get('morphology', 'Unknown')
            if morphology not in morphology_groups:
                morphology_groups[morphology] = []
            morphology_groups[morphology].append(node)

        # Draw nodes by morphology group with consistent colors
        for morphology, nodes in morphology_groups.items():
            # Use the predefined colors from self.morphology_colors_rgb
            color = self.morphology_colors_rgb.get(
                morphology, (0.5, 0.5, 0.5))  # Default to gray

            # Node size based on whether the cell divides
            node_sizes = [300 if G.nodes[n].get(
                'has_children', False) else 150 for n in nodes]

            # Draw this morphology group
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=nodes,
                                   node_size=node_sizes,
                                   node_color=[color] * len(nodes),
                                   alpha=0.8,
                                   label=morphology,
                                   ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        # Add a legend
        ax.legend(title=f"Morphology at {time_point} time")

        for morphology, nodes in morphology_groups.items():
            print(f"  {morphology}: {len(nodes)} cells")

    def calculate_diversity_metrics(self, tracks):
        """
        Calculate diversity metrics across the lineage tree.

        Parameters:
        -----------
        tracks : list
            List of track dictionaries

        Returns:
        --------
        dict
            Dictionary of diversity metrics
        """

        # Initialize counters
        internal_changes = 0
        total_cells_with_internal_data = 0
        generational_matches = 0
        total_parent_child_pairs = 0

        # Track morphology changes within cells (internal diversity)
        for track in tracks:
            if 't' in track and len(track['t']) > 1:
                first_morphology = self.get_morphology_at_time(
                    track, track['t'][0])
                last_morphology = self.get_morphology_at_time(
                    track, track['t'][-1])

                if first_morphology != last_morphology:
                    internal_changes += 1
                total_cells_with_internal_data += 1

        # Track morphology preservation across generations (robustness)
        for track in tracks:
            if 'children' in track and track['children']:
                parent_morphology = self.get_morphology_at_time(
                    track, track['t'][-1])

                for child_id in track['children']:
                    # Find child track
                    child_track = next(
                        (t for t in tracks if t['ID'] == child_id), None)
                    if child_track and 't' in child_track and len(child_track['t']) > 0:
                        child_first_morphology = self.get_morphology_at_time(
                            child_track, child_track['t'][0])

                        if parent_morphology == child_first_morphology:
                            generational_matches += 1
                        else:
                            # print(f"  Generational mismatch detected")
                            total_parent_child_pairs += 1

        # Calculate metrics
        internal_diversity = internal_changes / \
            total_cells_with_internal_data if total_cells_with_internal_data > 0 else 0
        robustness = generational_matches / \
            total_parent_child_pairs if total_parent_child_pairs > 0 else 0

        metrics = {
            "internal_diversity": internal_diversity,
            "robustness": robustness,
            "internal_changes": internal_changes,
            "total_cells_with_data": total_cells_with_internal_data,
            "generational_matches": generational_matches,
            "total_parent_child_pairs": total_parent_child_pairs
        }

        # print("Diversity metrics:")
        # print(f"  Internal diversity: {internal_diversity:.2f} ({internal_changes}/{total_cells_with_internal_data} cells changed)")
        # print(f"  Robustness: {robustness:.2f} ({generational_matches}/{total_parent_child_pairs} parent-child pairs matched)")

        return metrics

    def precompute_morphology_classifications(self, tracks):
        """
        Precomputes morphology classifications for all cells at all timepoints using
        the more reliable method and stores them in a lookup table.

        Parameters:
        -----------
        tracks : list
            List of track dictionaries with lineage information.

        Returns:
        --------
        dict
            Dictionary with structure {track_id: {timestamp: morphology_class}}
        """
        print("Precomputing morphology classifications for all cells...")

        # First, collect the overall morphology for each track using the more sophisticated method
        primary_morphology = self.collect_cell_morphology_data(tracks)

        # Now create a time-specific lookup dictionary
        morphology_by_time = {}

        for track in tracks:
            track_id = track['ID']
            if 't' not in track or not track['t']:
                continue

            morphology_by_time[track_id] = {}

            # Default morphology from the sophisticated method
            default_morphology = primary_morphology.get(track_id, "Healthy")

            # Handle special cases for specific timepoints
            timestamps = track['t']
            for i, t in enumerate(timestamps):
                # Special case for the final timepoint if the cell divides
                if i == len(timestamps) - 1 and 'children' in track and track['children']:
                    morphology_by_time[track_id][t] = "Divided"
                # Special case for elongated cells before division
                elif i >= len(timestamps) - 3 and 'children' in track and track['children']:
                    # Cells often elongate before dividing (in the last ~3 frames)
                    morphology_by_time[track_id][t] = "Elongated"
                # For deformed cells, maintain their classification throughout
                elif default_morphology == "Deformed":
                    morphology_by_time[track_id][t] = "Deformed"
                # For other timestamps, use the primary morphology
                else:
                    morphology_by_time[track_id][t] = default_morphology

        # Store for future use
        self.morphology_by_time = morphology_by_time

        return morphology_by_time

    def get_morphology_at_time(self, track, time):
        """
        Get the morphology of a cell at a specific time point using the precomputed lookup.

        Parameters:
        -----------
        track : dict
            Track dictionary with time information
        time : int
            Time point to get morphology for

        Returns:
        --------
        str
            Morphology class at that time
        """
        track_id = track['ID']

        # Use the precomputed lookup if available
        if hasattr(self, 'morphology_by_time') and track_id in self.morphology_by_time and time in self.morphology_by_time[track_id]:
            morphology = self.morphology_by_time[track_id][time]
            return morphology

        # If not in lookup, fall back to cell_mapping if available
        if hasattr(self, 'cell_mapping') and self.cell_mapping:
            cell_id = None
            for cid, cell_data in self.cell_mapping.items():
                if ('track_id' in cell_data and cell_data['track_id'] == track_id and
                        'time' in cell_data and cell_data['time'] == time):
                    cell_id = cid
                    break

            if cell_id and cell_id in self.cell_mapping and 'metrics' in self.cell_mapping[cell_id]:
                morphology = self.cell_mapping[cell_id]['metrics'].get(
                    'morphology_class', 'Unknown')
                return morphology

        # If still not found, use the more sophisticated fallback
        if hasattr(self, 'collect_cell_morphology_data'):
            # Get overall morphology using the better method
            all_morphologies = self.collect_cell_morphology_data([track])
            if track_id in all_morphologies:
                morphology = all_morphologies[track_id]
                return morphology

        # Last resort - basic inference from track properties
        if 'children' in track and track['children']:
            if time == track['t'][-1]:
                return "Divided"
            elif 't' in track and time >= track['t'][-3]:
                return "Elongated"

        # Check for deformed based on track properties
        if 'x' in track and 'y' in track and len(track['x']) > 3:
            positions = list(zip(track['x'], track['y']))
            # Calculate path straightness
            direct_distance = ((positions[-1][0] - positions[0][0])**2 +
                               (positions[-1][1] - positions[0][1])**2)**0.5
            path_length = 0
            for i in range(1, len(positions)):
                segment_length = ((positions[i][0] - positions[i-1][0])**2 +
                                  (positions[i][1] - positions[i-1][1])**2)**0.5
                path_length += segment_length

            straightness = direct_distance / path_length if path_length > 0 else 1

            if straightness < 0.6:
                return "Deformed"

        return "Healthy"

    def export_morphology_classifications(self, parent_widget=None):
        """
        Exports the precomputed morphology classifications to a CSV file.

        Parameters:
        -----------
        parent_widget : QWidget
            Parent widget for the file dialog
        """
        # Check if we have precomputed morphologies
        if not hasattr(self, "morphology_by_time") or not self.morphology_by_time:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                parent_widget,
                "Error",
                "No precomputed morphology data available. Run visualization first.")
            return

        # Ask user for save location
        from PySide6.QtWidgets import QFileDialog
        save_path, _ = QFileDialog.getSaveFileName(
            parent_widget, "Save Morphology Classification Data", "", "CSV Files (*.csv)")

        if not save_path:
            return

        try:
            # Create DataFrame from the morphology_by_time dictionary
            import pandas as pd

            # Flatten the nested dictionary into rows
            rows = []
            for track_id, time_dict in self.morphology_by_time.items():
                for time, morphology in time_dict.items():
                    rows.append({
                        'track_id': track_id,
                        'time': time,
                        'morphology': morphology
                    })

            # Convert to DataFrame
            morphology_df = pd.DataFrame(rows)

            # Sort by track_id and time for readability
            morphology_df = morphology_df.sort_values(['track_id', 'time'])

            # Save to CSV
            morphology_df.to_csv(save_path, index=False)

            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                parent_widget,
                "Export Complete",
                f"Morphology classification data saved to:\n{save_path}")

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                parent_widget,
                "Export Error",
                f"Failed to export morphology data: {str(e)}")


    def calculate_growth_and_division_metrics(self, tracks):
        """
        Calculate growth rate and division timing from tracking data.
        
        Parameters:
        -----------
        tracks : list
            List of track dictionaries with lineage information.
            
        Returns:
        --------
        dict
            Dictionary containing growth and division metrics.
        """
        division_times = []
        growth_rates = []
        
        for track in tracks:
            # Skip tracks that don't divide
            if 'children' not in track or not track['children']:
                continue
                
            # Get timestamps
            if 't' in track and len(track['t']) > 0:
                t_appearance = track['t'][0]
                t_division = track['t'][-1]
                
                # Calculate division timing
                dt = t_division - t_appearance
                
                # Skip very short tracks that might be tracking errors
                if dt <= 0:
                    continue
                    
                division_times.append(dt)
                
                # Calculate growth rate based on division time
                # (assuming exponential growth with doubling)
                growth_rate = np.log(2) / dt
                growth_rates.append(growth_rate)
        
        # Calculate statistics
        if division_times:
            avg_division_time = np.mean(division_times)
            std_division_time = np.std(division_times)
            median_division_time = np.median(division_times)
            
            avg_growth_rate = np.mean(growth_rates)
            std_growth_rate = np.std(growth_rates)
        else:
            avg_division_time = std_division_time = median_division_time = 0
            avg_growth_rate = std_growth_rate = 0
        
        # Compile results
        results = {
            'division_times': division_times,
            'growth_rates': growth_rates,
            'avg_division_time': avg_division_time,
            'std_division_time': std_division_time,
            'median_division_time': median_division_time,
            'avg_growth_rate': avg_growth_rate,
            'std_growth_rate': std_growth_rate,
            'total_dividing_cells': len(division_times)
        }
        
        return results
    
    
    
    def visualize_growth_and_division(self, tracks, growth_metrics):
        """
        Create visualizations for growth and division timing data.
        
        Parameters:
        -----------
        tracks : list
            List of track dictionaries with lineage information.
        growth_metrics : dict
            Output from calculate_growth_and_division_metrics function.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Histogram of division times
        axes[0, 0].hist(growth_metrics['division_times'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Division Time Distribution')
        axes[0, 0].set_xlabel('Time (frames)')
        axes[0, 0].set_ylabel('Cell Count')
        axes[0, 0].axvline(growth_metrics['avg_division_time'], color='red', 
                        linestyle='--', label=f"Mean: {growth_metrics['avg_division_time']:.1f}")
        axes[0, 0].axvline(growth_metrics['median_division_time'], color='green', 
                        linestyle='--', label=f"Median: {growth_metrics['median_division_time']:.1f}")
        axes[0, 0].legend()
        
        # 2. Histogram of growth rates
        axes[0, 1].hist(growth_metrics['growth_rates'], bins=20, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Growth Rate Distribution')
        axes[0, 1].set_xlabel('Growth Rate (ln(2)/division time)')
        axes[0, 1].set_ylabel('Cell Count')
        axes[0, 1].axvline(growth_metrics['avg_growth_rate'], color='red', 
                        linestyle='--', label=f"Mean: {growth_metrics['avg_growth_rate']:.4f}")
        axes[0, 1].legend()
        
        # 3. Division time by generation
        # Create a lookup from track_id to division time
        division_time_by_id = {}
        for track in tracks:
            if 'children' in track and track['children'] and 't' in track and len(track['t']) > 0:
                dt = track['t'][-1] - track['t'][0]
                if dt > 0:
                    division_time_by_id[track['ID']] = dt
        
        # Calculate parent-child division time pairs
        parent_child_division_pairs = []
        for track in tracks:
            if 'children' in track and track['children'] and track['ID'] in division_time_by_id:
                parent_dt = division_time_by_id[track['ID']]
                
                for child_id in track['children']:
                    if child_id in division_time_by_id:
                        child_dt = division_time_by_id[child_id]
                        parent_child_division_pairs.append((parent_dt, child_dt))
        
        # Plot parent vs child division times
        if parent_child_division_pairs:
            parent_dts, child_dts = zip(*parent_child_division_pairs)
            axes[1, 0].scatter(parent_dts, child_dts, alpha=0.7)
            axes[1, 0].set_title('Division Time: Parent vs. Child')
            axes[1, 0].set_xlabel('Parent Division Time')
            axes[1, 0].set_ylabel('Child Division Time')
            
            # Add y=x reference line
            min_val = min(min(parent_dts), min(child_dts))
            max_val = max(max(parent_dts), max(child_dts))
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Calculate correlation
            correlation = np.corrcoef(parent_dts, child_dts)[0, 1]
            axes[1, 0].text(0.05, 0.95, f"Correlation: {correlation:.2f}", 
                        transform=axes[1, 0].transAxes, 
                        verticalalignment='top')
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary = (
            f"Growth & Division Summary\n\n"
            f"Total dividing cells: {growth_metrics['total_dividing_cells']}\n\n"
            f"Division Time:\n"
            f"  Mean: {growth_metrics['avg_division_time']:.1f} frames\n"
            f"  Std Dev: {growth_metrics['std_division_time']:.1f} frames\n"
            f"  Median: {growth_metrics['median_division_time']:.1f} frames\n\n"
            f"Growth Rate:\n"
            f"  Mean: {growth_metrics['avg_growth_rate']:.4f}\n"
            f"  Std Dev: {growth_metrics['std_growth_rate']:.4f}\n"
        )
        
        if parent_child_division_pairs:
            summary += f"\nParent-Child Division Time Correlation: {correlation:.2f}"
        
        axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', horizontalalignment='left',
                    fontfamily='monospace')
        
        plt.tight_layout()
        return fig
    
    
    
    def calculate_division_robustness(tracks):
        """
        Calculate robustness metrics comparing division timing across generations.
        
        Parameters:
        -----------
        tracks : list
            List of track dictionaries with lineage information.
            
        Returns:
        --------
        dict
            Dictionary containing robustness metrics.
        """
        # Create a lookup from track_id to division time
        division_time_by_id = {}
        for track in tracks:
            if 'children' in track and track['children'] and 't' in track and len(track['t']) > 0:
                dt = track['t'][-1] - track['t'][0]
                if dt > 0:
                    division_time_by_id[track['ID']] = dt
        
        # Calculate parent-child division time pairs
        parent_child_division_pairs = []
        for track in tracks:
            if 'children' in track and track['children'] and track['ID'] in division_time_by_id:
                parent_dt = division_time_by_id[track['ID']]
                
                for child_id in track['children']:
                    if child_id in division_time_by_id:
                        child_dt = division_time_by_id[child_id]
                        parent_child_division_pairs.append((parent_dt, child_dt))
        
        if not parent_child_division_pairs:
            return {
                'robustness_score': 0,
                'correlation': 0,
                'mean_relative_error': 0,
                'pairs_analyzed': 0
            }
        
        # Unzip parent-child pairs
        parent_dts, child_dts = zip(*parent_child_division_pairs)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(parent_dts, child_dts)[0, 1]
        
        # Calculate mean relative error
        relative_errors = [abs(c - p) / p for p, c in parent_child_division_pairs]
        mean_relative_error = np.mean(relative_errors)
        
        # Calculate robustness score (1 = perfect robustness, 0 = no robustness)
        robustness_score = 1 - mean_relative_error
        
        return {
            'robustness_score': robustness_score,
            'correlation': correlation,
            'mean_relative_error': mean_relative_error,
            'pairs_analyzed': len(parent_child_division_pairs)
        }