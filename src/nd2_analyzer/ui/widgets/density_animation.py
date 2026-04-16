from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSlider, QCheckBox, QProgressBar, QComboBox,
                               QWidget, QFrame, QApplication, QMessageBox, QFileDialog,
                               QButtonGroup, QRadioButton, QGroupBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2
from pathlib import Path
import imageio
from datetime import datetime


class DensityAnimationGenerator(QThread):
    """Generate animation frames from real bacterial tracking data"""
    frame_ready = Signal(int, dict)  # frame_number, frame_data_dict
    progress_updated = Signal(int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, tracks, chamber_dimensions, grid_size=50):
        super().__init__()
        self.tracks = tracks
        self.chamber_dimensions = chamber_dimensions
        self.grid_size = grid_size
        self.is_cancelled = False

    def run(self):
        try:
            self.generate_frames_from_tracks()
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self.is_cancelled = True

    def generate_frames_from_tracks(self):
        """Generate animation frames from real bacterial tracking data"""
        # Get all unique time points from tracks
        all_time_points = set()
        for track in self.tracks:
            if 't' in track:
                all_time_points.update(track['t'])
            else:
                # If no time data, assume sequential frames
                all_time_points.update(range(len(track.get('x', []))))

        if not all_time_points:
            self.error.emit("No time data found in tracks")
            return

        time_points = sorted(all_time_points)

        # Generate frame for each time point
        for i, t in enumerate(time_points):
            if self.is_cancelled:
                return

            frame_data = self.create_frame_at_time(t, time_points[:i+1])
            self.frame_ready.emit(i, frame_data)

            progress = int((i + 1) / len(time_points) * 100)
            self.progress_updated.emit(progress)

        self.finished.emit()

    def create_frame_at_time(self, time_point, all_previous_times):
        """Create frame data for a specific time point with both live and cumulative data"""
        width, height = self.chamber_dimensions

        # LIVE POSITIONS: Only cells at current time point
        live_positions = []
        live_velocities = []
        active_tracks = []

        for track in self.tracks:
            track_times = track.get('t', list(range(len(track.get('x', [])))))

            # Check if this track has data at current time point
            if time_point in track_times:
                time_idx = track_times.index(time_point)

                x = track['x'][time_idx]
                y = track['y'][time_idx]
                live_positions.append((x, y))

                # Calculate velocity if possible
                velocity = 0
                if time_idx > 0:
                    prev_x = track['x'][time_idx - 1]
                    prev_y = track['y'][time_idx - 1]
                    dt = track_times[time_idx] - track_times[time_idx - 1]
                    if dt > 0:
                        dx = x - prev_x
                        dy = y - prev_y
                        velocity = np.sqrt(dx**2 + dy**2) / dt

                live_velocities.append(velocity)
                active_tracks.append({
                    'id': track.get('ID', -1),
                    'x': x,
                    'y': y,
                    'velocity': velocity
                })

        # CUMULATIVE POSITIONS: All positions from start up to current time
        cumulative_positions = []
        for track in self.tracks:
            track_times = track.get('t', list(range(len(track.get('x', [])))))

            # Get ALL positions from start up to current time point
            for i, t in enumerate(track_times):
                if t <= time_point:  # Include all times up to current
                    x = track['x'][i]
                    y = track['y'][i]
                    cumulative_positions.append((x, y))

        # Create density grids for both modes
        grid_x_bins = np.arange(0, width + self.grid_size, self.grid_size)
        grid_y_bins = np.arange(0, height + self.grid_size, self.grid_size)

        # Live density grid
        if live_positions:
            x_coords, y_coords = zip(*live_positions)
            live_density_grid, _, _ = np.histogram2d(
                y_coords, x_coords,
                bins=[grid_y_bins, grid_x_bins]
            )
        else:
            live_density_grid = np.zeros(
                (len(grid_y_bins)-1, len(grid_x_bins)-1))

        # Cumulative density grid
        if cumulative_positions:
            x_coords_cum, y_coords_cum = zip(*cumulative_positions)
            cumulative_density_grid, _, _ = np.histogram2d(
                y_coords_cum, x_coords_cum,
                bins=[grid_y_bins, grid_x_bins]
            )
        else:
            cumulative_density_grid = np.zeros(
                (len(grid_y_bins)-1, len(grid_x_bins)-1))

        # Calculate statistics for both modes
        live_stats = {
            'total_cells': len(live_positions),
            'avg_velocity': np.mean(live_velocities) if live_velocities else 0,
            'max_velocity': np.max(live_velocities) if live_velocities else 0,
            'time_point': time_point,
            'high_density_regions': np.sum(live_density_grid > 10),
            'max_density': np.max(live_density_grid),
            'active_tracks': len(active_tracks)
        }

        cumulative_stats = {
            'total_positions': len(cumulative_positions),
            # Use live velocity for current frame
            'avg_velocity': np.mean(live_velocities) if live_velocities else 0,
            'max_velocity': np.max(live_velocities) if live_velocities else 0,
            'time_point': time_point,
            'high_density_regions': np.sum(cumulative_density_grid > 10),
            'max_density': np.max(cumulative_density_grid),
            'active_tracks': len(active_tracks)
        }

        return {
            'live_positions': live_positions,
            'cumulative_positions': cumulative_positions,
            'live_density_grid': live_density_grid,
            'cumulative_density_grid': cumulative_density_grid,
            'velocities': live_velocities,
            'tracks': active_tracks,
            'live_stats': live_stats,
            'cumulative_stats': cumulative_stats,
            'grid_x_bins': grid_x_bins,
            'grid_y_bins': grid_y_bins
        }


class BacterialDensityAnimationModal(QDialog):
    """Modal dialog for bacterial density animation"""

    def __init__(self, motility_dialog, parent=None):
        super().__init__(parent)
        self.motility_dialog = motility_dialog
        self.animation_frames = []
        self.current_frame = 0
        self.is_playing = False
        self.fps = 3

        self.setWindowTitle("Cell Density Animation")
        self.setModal(True)
        self.resize(1600, 1000)

        # Track selection
        self.use_filtered_tracks = True

        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)

        # Header with controls
        header_layout = self.create_header()
        layout.addLayout(header_layout)

        # Main content
        content_layout = QHBoxLayout()

        # Visualization area
        viz_layout = QVBoxLayout()

        # Canvas
        self.figure = Figure(figsize=(14, 10))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(True)
        viz_layout.addWidget(self.progress_bar)

        content_layout.addLayout(viz_layout, 3)

        # Controls panel
        controls_panel = self.create_controls_panel()
        content_layout.addWidget(controls_panel, 1)

        layout.addLayout(content_layout)

        # Timeline controls
        timeline_layout = self.create_timeline_controls()
        layout.addLayout(timeline_layout)

    def create_header(self):
        """Create header with data source selection"""
        header_layout = QHBoxLayout()

        # Title
        title_label = QLabel("Cell Density Formation")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Data source selection
        data_group = QGroupBox("Data Source")
        data_layout = QHBoxLayout(data_group)

        # Get counts from motility dialog
        filtered_count = len(
            getattr(self.motility_dialog, 'tracked_cells', []))
        all_count = len(getattr(self.motility_dialog, 'lineage_tracks', []))

        self.filtered_radio = QRadioButton(
            f"Filtered Tracks ({filtered_count})")
        self.all_radio = QRadioButton(f"All Tracks ({all_count})")

        self.filtered_radio.setChecked(True)
        self.filtered_radio.toggled.connect(self.on_data_source_changed)

        data_layout.addWidget(self.filtered_radio)
        data_layout.addWidget(self.all_radio)

        header_layout.addWidget(data_group)

        # Generate button
        self.generate_btn = QPushButton("🎬 Generate Animation")
        self.generate_btn.clicked.connect(self.start_animation_generation)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B35;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #E55A2B; }
            QPushButton:pressed { background-color: #CC4B21; }
        """)
        header_layout.addWidget(self.generate_btn)

        # Export button
        self.export_btn = QPushButton("📁 Export")
        self.export_btn.clicked.connect(self.export_animation)
        self.export_btn.setEnabled(False)
        header_layout.addWidget(self.export_btn)

        return header_layout

    def create_controls_panel(self):
        """Create controls panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)

        # Animation mode selection
        mode_group = QGroupBox("Animation Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.live_mode_rb = QRadioButton("Live Bacteria")
        self.live_mode_rb.setToolTip(
            "Show only bacteria alive at current time point")
        self.cumulative_mode_rb = QRadioButton("Cumulative Footprint")
        self.cumulative_mode_rb.setToolTip(
            "Show all cell positions from start to current time")

        self.live_mode_rb.setChecked(True)

        # Connect mode change to update display
        self.live_mode_rb.toggled.connect(self.on_mode_changed)

        mode_layout.addWidget(self.live_mode_rb)
        mode_layout.addWidget(self.cumulative_mode_rb)
        layout.addWidget(mode_group)

        # Live cell counter - BIG and prominent
        counter_label = QLabel("CELL COUNT")
        counter_label.setFont(QFont("Arial", 12, QFont.Bold))
        counter_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(counter_label)

        self.cell_counter = QLabel("0")
        self.cell_counter.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: #2E86AB;
                background-color: white;
                border: 3px solid #2E86AB;
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
            }
        """)
        self.cell_counter.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cell_counter)

        # Mode description
        self.mode_description = QLabel("Live: bacteria at current time")
        self.mode_description.setStyleSheet(
            "QLabel { font-size: 10px; color: #666; text-align: center; }")
        self.mode_description.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.mode_description)

        # Animation settings
        settings_label = QLabel("Animation Settings")
        settings_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(settings_label)

        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setMinimum(1)
        self.fps_slider.setMaximum(10)
        self.fps_slider.setValue(3)
        self.fps_slider.valueChanged.connect(self.update_fps)
        speed_layout.addWidget(self.fps_slider)
        self.fps_label = QLabel("3 fps")
        speed_layout.addWidget(self.fps_label)
        layout.addLayout(speed_layout)

        # Visualization layers
        layers_label = QLabel("Display Layers")
        layers_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(layers_label)

        self.show_density_cb = QCheckBox("Density Heatmap")
        self.show_density_cb.setChecked(True)
        self.show_density_cb.toggled.connect(self.on_display_changed)
        layout.addWidget(self.show_density_cb)

        self.show_cells_cb = QCheckBox("Individual Bacteria")
        self.show_cells_cb.setChecked(True)
        self.show_cells_cb.toggled.connect(self.on_display_changed)
        layout.addWidget(self.show_cells_cb)

        self.show_grid_cb = QCheckBox("Analysis Grid")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.toggled.connect(self.on_display_changed)
        layout.addWidget(self.show_grid_cb)

        # Current frame statistics
        layout.addWidget(QLabel("Frame Statistics"))
        self.stats_label = QLabel("Generate animation to see stats...")
        self.stats_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.stats_label)

        layout.addStretch()
        return panel

    def create_timeline_controls(self):
        """Create timeline controls"""
        layout = QHBoxLayout()

        # Play/pause
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        layout.addWidget(self.play_btn)

        # Timeline slider
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.valueChanged.connect(self.seek_frame)
        layout.addWidget(self.timeline_slider)

        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        layout.addWidget(self.time_label)

        return layout

    def on_data_source_changed(self):
        """Handle data source selection change"""
        self.use_filtered_tracks = self.filtered_radio.isChecked()
        # Reset animation if already generated
        if self.animation_frames:
            self.animation_frames = []
            self.play_btn.setEnabled(False)
            self.export_btn.setEnabled(False)

    def on_mode_changed(self):
        """Handle animation mode change"""
        if self.live_mode_rb.isChecked():
            self.mode_description.setText("Live: bacteria at current time")
        else:
            self.mode_description.setText(
                "Cumulative: all positions up to current time")

        # Refresh current frame if animation is loaded
        if self.animation_frames:
            self.display_frame(self.current_frame)

    def on_display_changed(self):
        """Handle display layer changes"""
        # Refresh current frame if animation is loaded
        if self.animation_frames:
            self.display_frame(self.current_frame)

    def start_animation_generation(self):
        """Start generating animation from real tracking data"""
        # Get the appropriate tracks
        if self.use_filtered_tracks:
            tracks = getattr(self.motility_dialog, 'tracked_cells', [])
            source_name = "filtered"
        else:
            tracks = getattr(self.motility_dialog, 'lineage_tracks', [])
            source_name = "all"

        if not tracks:
            QMessageBox.warning(
                self, "No Data", f"No {source_name} tracks available")
            return

        # Get chamber dimensions
        chamber_dims = getattr(self.motility_dialog,
                               'chamber_dimensions', (1392, 1040))

        # Start background generation
        self.generator = DensityAnimationGenerator(tracks, chamber_dims)
        self.generator.frame_ready.connect(self.add_frame)
        self.generator.progress_updated.connect(self.progress_bar.setValue)
        self.generator.finished.connect(self.generation_finished)
        self.generator.error.connect(self.generation_error)

        # Update UI
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.generator.start()

    def add_frame(self, frame_idx, frame_data):
        """Add a generated frame"""
        self.animation_frames.append(frame_data)

        # Show first frame immediately
        if frame_idx == 0:
            self.display_frame(0)

    def generation_finished(self):
        """Handle completion of animation generation"""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        # Update timeline
        if self.animation_frames:
            self.timeline_slider.setMaximum(len(self.animation_frames) - 1)
            total_seconds = len(self.animation_frames) / self.fps
            self.time_label.setText(
                f"00:00 / {int(total_seconds//60):02d}:{int(total_seconds%60):02d}")


    def display_frame(self, frame_idx):
        """Display frame from real data"""
        if not self.animation_frames or frame_idx >= len(self.animation_frames):
            return

        frame_data = self.animation_frames[frame_idx]

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Get chamber dimensions
        chamber_dims = getattr(self.motility_dialog,
                               'chamber_dimensions', (1392, 1040))
        width, height = chamber_dims

        # Choose data based on mode
        if self.live_mode_rb.isChecked():
            positions = frame_data['live_positions']
            density_grid = frame_data['live_density_grid']
            stats = frame_data['live_stats']
            mode_label = "Live"
            cell_count = stats['total_cells']
        else:
            positions = frame_data['cumulative_positions']
            density_grid = frame_data['cumulative_density_grid']
            stats = frame_data['cumulative_stats']
            mode_label = "Cumulative"
            cell_count = stats['total_positions']

        # 1. Density heatmap
        if self.show_density_cb.isChecked():
            x_bins = frame_data['grid_x_bins']
            y_bins = frame_data['grid_y_bins']

            if np.max(density_grid) > 0:
                im = ax.imshow(
                    density_grid,
                    extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                    origin='lower',
                    cmap='plasma',
                    alpha=0.7,
                    vmin=0,
                    vmax=max(20, np.max(density_grid))
                )

        # 2. Individual bacteria
        if self.show_cells_cb.isChecked() and positions:
            x_coords, y_coords = zip(*positions)
            # Use different alpha for cumulative mode to show overlapping
            alpha = 0.4 if self.cumulative_mode_rb.isChecked() else 0.8
            size = 8 if self.cumulative_mode_rb.isChecked() else 12
            ax.scatter(x_coords, y_coords, s=size, color='cyan', alpha=alpha,
                       edgecolors='blue', linewidth=0.3)

        # 3. Grid overlay
        if self.show_grid_cb.isChecked():
            x_bins = frame_data['grid_x_bins']
            y_bins = frame_data['grid_y_bins']

            for x in x_bins[::2]:  # Every other line
                ax.axvline(x, color='white', alpha=0.3, linewidth=0.5)
            for y in y_bins[::2]:
                ax.axhline(y, color='white', alpha=0.3, linewidth=0.5)

        # 4. Display count prominently on plot
        count_text = f'{mode_label}: {cell_count}'
        ax.text(0.02, 0.98, count_text,
                transform=ax.transAxes,
                fontsize=20, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                verticalalignment='top')

        # Set limits and labels
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')

        # Title with time info
        title = f"Cell Density Formation - Time Point {stats['time_point']} ({mode_label} Mode)"
        ax.set_title(title, fontsize=14, fontweight='bold')

        self.canvas.draw()
        self.update_frame_stats(stats, cell_count, mode_label)

    def update_frame_stats(self, stats, cell_count, mode_label):
        """Update statistics display"""
        # Update big cell counter
        self.cell_counter.setText(str(cell_count))

        # Update detailed stats
        count_label = "Live Bacteria" if mode_label == "Live" else "Total Positions"
        stats_text = f"""Frame: {self.current_frame + 1}/{len(self.animation_frames)}
            Time Point: {stats['time_point']}

            {count_label}: {cell_count}
            Avg Velocity: {stats['avg_velocity']:.2f} px/frame
            Max Velocity: {stats['max_velocity']:.2f} px/frame
            High Density Regions: {stats['high_density_regions']}
            Max Density: {stats['max_density']:.0f} cells/block"""

        self.stats_label.setText(stats_text)

    def setup_timer(self):
        """Setup animation timer"""
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.next_frame)

    def toggle_playback(self):
        """Toggle animation playback"""
        if not self.animation_frames:
            return

        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_btn.setText("⏸ Pause")
            self.animation_timer.start(1000 // self.fps)
        else:
            self.play_btn.setText("▶ Play")
            self.animation_timer.stop()

    def next_frame(self):
        """Advance to next frame"""
        if not self.animation_frames:
            return

        self.current_frame = (self.current_frame +
                              1) % len(self.animation_frames)
        self.timeline_slider.setValue(self.current_frame)
        self.display_frame(self.current_frame)

        # Update time
        current_seconds = self.current_frame / self.fps
        total_seconds = len(self.animation_frames) / self.fps
        self.time_label.setText(f"{int(current_seconds//60):02d}:{int(current_seconds%60):02d} / "
                                f"{int(total_seconds//60):02d}:{int(total_seconds%60):02d}")

    def seek_frame(self, frame_idx):
        """Seek to specific frame"""
        if not self.animation_frames:
            return

        self.current_frame = frame_idx
        self.display_frame(frame_idx)

        current_seconds = frame_idx / self.fps
        total_seconds = len(self.animation_frames) / self.fps
        self.time_label.setText(f"{int(current_seconds//60):02d}:{int(current_seconds%60):02d} / "
                                f"{int(total_seconds//60):02d}:{int(total_seconds%60):02d}")

    def update_fps(self, fps):
        """Update FPS"""
        self.fps = fps
        self.fps_label.setText(f"{fps} fps")

        if self.is_playing:
            self.animation_timer.stop()
            self.animation_timer.start(1000 // self.fps)

    def generation_error(self, error_msg):
        """Handle generation errors"""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Generation Error",
                             f"Failed to generate animation: {error_msg}")

    def export_animation(self):
        """Export animation"""
        if not self.animation_frames:
            QMessageBox.warning(self, "Export Error", "No animation to export")
            return
            
        source_type = "filtered" if self.use_filtered_tracks else "all"
        mode_type = "live" if self.live_mode_rb.isChecked() else "cumulative"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"bacterial_density_{mode_type}_{source_type}_{timestamp}"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Animation", default_name,
            "MP4 Video (*.mp4);;Animated GIF (*.gif)")
            
        if not file_path:
            return
            
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Store current mode for export
            export_live_mode = self.live_mode_rb.isChecked()
            
            # Create frames for export
            rendered_frames = []
            for i, frame_data in enumerate(self.animation_frames):
                # Create figure for export
                fig = plt.figure(figsize=(12, 8), dpi=100)
                ax = fig.add_subplot(111)
                
                chamber_dims = getattr(self.motility_dialog, 'chamber_dimensions', (1392, 1040))
                width, height = chamber_dims
                
                # Choose data based on export mode
                if export_live_mode:
                    positions = frame_data['live_positions']
                    density_grid = frame_data['live_density_grid']
                    stats = frame_data['live_stats']
                    mode_label = "Live"
                    cell_count = stats['total_cells']
                else:
                    positions = frame_data['cumulative_positions']
                    density_grid = frame_data['cumulative_density_grid']
                    stats = frame_data['cumulative_stats']
                    mode_label = "Cumulative"
                    cell_count = stats['total_positions']
                
                # Render density and cells
                if self.show_density_cb.isChecked():
                    x_bins = frame_data['grid_x_bins']
                    y_bins = frame_data['grid_y_bins']
                    
                    if np.max(density_grid) > 0:
                        ax.imshow(density_grid, extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], 
                                origin='lower', cmap='plasma', alpha=0.7)
                
                if self.show_cells_cb.isChecked() and positions:
                    x_coords, y_coords = zip(*positions)
                    alpha = 0.4 if not export_live_mode else 0.8
                    size = 6 if not export_live_mode else 8
                    ax.scatter(x_coords, y_coords, s=size, color='cyan', alpha=alpha)
                
                # Add count
                ax.text(0.02, 0.98, f'{mode_label}: {cell_count}', transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
                
                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
                ax.set_title(f"Cell Density - Time Point {stats['time_point']} ({mode_label})")
                
                import matplotlib
                matplotlib.use('Agg')  
                
                fig.canvas.draw()
                width, height = fig.get_size_inches() * fig.dpi
                width, height = int(width), int(height)

                # Get the RGBA buffer
                canvas = fig.canvas
                buf = canvas.buffer_rgba()
                img_array = np.asarray(buf).reshape((height, width, 4))

                # Convert RGBA to RGB
                img_array = img_array[:, :, :3]
                rendered_frames.append(img_array)
                
                plt.close(fig)
                
                progress = int((i + 1) / len(self.animation_frames) * 100)
                self.progress_bar.setValue(progress)
                QApplication.processEvents()
            
            # Save animation
            if file_path.lower().endswith('.gif'):
                imageio.mimsave(file_path, rendered_frames, fps=self.fps)
            else:
                if not file_path.lower().endswith('.mp4'):
                    file_path += '.mp4'
                imageio.mimsave(file_path, rendered_frames, fps=self.fps, codec='libx264')
            
            self.progress_bar.setVisible(False)
            QMessageBox.information(self, "Export Complete", f"Animation exported to {file_path}")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")


def add_animation_button_to_motility_widget(motility_dialog):
    """Add animation button to motility widget"""

    def open_bacterial_animation():
        """Open bacterial animation modal"""
        try:
            animation_modal = BacterialDensityAnimationModal(motility_dialog)
            animation_modal.exec()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(motility_dialog, "Animation Error",
                                 f"Failed to open animation: {str(e)}")

    # Add button to density layout
    if hasattr(motility_dialog, 'density_layout'):
        animation_btn = QPushButton("🎬 Animate Cell Density Formation")
        animation_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF6B35;
                    color: white;
                    border: none;
                    padding: 15px 30px;
                    border-radius: 10px;
                    font-weight: bold;
                    font-size: 15px;
                }
                QPushButton:hover {
                    background-color: #E55A2B;
                }
                QPushButton:pressed {
                    background-color: #CC4B21;
                }
            """)
        animation_btn.clicked.connect(open_bacterial_animation)

        # Insert at top of density layout
        motility_dialog.density_layout.insertWidget(0, animation_btn)

        return animation_btn
    else:
        print("Warning: Could not find density_layout in motility dialog")
        return None
