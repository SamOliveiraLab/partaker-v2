# lineage_dialog.py
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QRadioButton, QButtonGroup, QTabWidget,
                             QWidget, QFileDialog, QMessageBox, QProgressDialog)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
from pubsub import pub

class LineageDialog(QDialog):
    """
    Dialog for visualizing cell lineage trees and relationships.
    """
    
    def __init__(self, lineage_tracks, parent=None):
        super().__init__(parent)
        
        self.lineage_tracks = lineage_tracks
        self.current_tree_index = 0
        self.available_trees = []
        
        # Set dialog properties
        self.setWindowTitle("Cell Lineage Visualization")
        self.setMinimumWidth(800)
        self.setMinimumHeight(700)
        
        # Initialize UI
        self.init_ui()
        
        # Generate initial visualization
        self.generate_visualization()
    
    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Visualization type selection
        viz_layout = QHBoxLayout()
        viz_layout.addWidget(QLabel("Visualization Style:"))
        self.viz_type = QComboBox()
        self.viz_type.addItems(["Standard Lineage Tree", "Morphology-Enhanced Tree"])
        self.viz_type.currentIndexChanged.connect(self.generate_visualization)
        viz_layout.addWidget(self.viz_type)
        layout.addLayout(viz_layout)
        
        # Tree selection options
        selection_layout = QHBoxLayout()
        option_group = QButtonGroup(self)
        
        self.top_radio = QRadioButton("Top 5 Largest Lineage Trees")
        self.top_radio.setChecked(True)
        option_group.addButton(self.top_radio)
        selection_layout.addWidget(self.top_radio)
        
        self.cell_radio = QRadioButton("Specific Cell Lineage:")
        option_group.addButton(self.cell_radio)
        selection_layout.addWidget(self.cell_radio)
        
        self.cell_combo = QComboBox()
        self.cell_combo.setEnabled(False)
        selection_layout.addWidget(self.cell_combo)
        
        # Populate cell dropdown
        dividing_cells = [track['ID'] for track in self.lineage_tracks 
                          if track.get('children', [])]
        dividing_cells.sort()
        for cell_id in dividing_cells:
            self.cell_combo.addItem(f"Cell {cell_id}")
        
        layout.addLayout(selection_layout)
        
        # Connect radio button signals
        self.top_radio.toggled.connect(self.update_combo_state)
        self.cell_radio.toggled.connect(self.update_combo_state)
        
        # Canvas for visualization
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Tree navigation controls
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous Tree")
        self.prev_button.clicked.connect(self.previous_tree)
        
        self.tree_counter = QLabel("Tree 1/1")
        self.tree_counter.setAlignment(Qt.AlignCenter)
        
        self.next_button = QPushButton("Next Tree")
        self.next_button.clicked.connect(self.next_tree)
        
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.tree_counter)
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.view_button = QPushButton("Refresh View")
        self.view_button.clicked.connect(self.generate_visualization)
        
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_visualization)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        self.time_comp_button = QPushButton("Time Zero vs Time Last")
        self.time_comp_button.clicked.connect(self.show_time_comparison)
        
        button_layout.addWidget(self.view_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.time_comp_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def update_combo_state(self):
        """Update the cell combobox state based on radio selection"""
        self.cell_combo.setEnabled(self.cell_radio.isChecked())
    
    def generate_visualization(self):
        """Generate lineage tree visualization"""
        # This is just the core logic - actual visualization would use LineageVisualization
        # Clear figure
        self.figure.clear()
        
        # Get root cell ID
        root_cell_id = self.get_root_cell_id()
        if root_cell_id is None:
            return
            
        # Create visualization
        if self.viz_type.currentText() == "Standard Lineage Tree":
            # Call the standard visualization function
            from lineage_visualization import LineageVisualization
            visualizer = LineageVisualization()
            visualizer.create_lineage_tree(self.lineage_tracks, self.canvas, root_cell_id)
        else:
            # Call the morphology-enhanced visualization function
            from lineage_visualization import LineageVisualization
            visualizer = LineageVisualization()
            visualizer.visualize_morphology_lineage_tree(self.lineage_tracks, self.canvas, root_cell_id)
        
        # Update UI
        self.update_ui_for_current_tree()
    
    def get_root_cell_id(self):
        """Determine the root cell ID for visualization based on selection"""
        if self.cell_radio.isChecked():
            # Use selected cell
            text = self.cell_combo.currentText()
            if text:
                return int(text.replace("Cell ", ""))
            return None
        else:
            # Use top tree mode - find connected components
            import networkx as nx
            G = nx.DiGraph()
            
            # Build graph
            for track in self.lineage_tracks:
                G.add_node(track['ID'])
                if 'children' in track and track['children']:
                    for child_id in track['children']:
                        G.add_edge(track['ID'], child_id)
            
            # Find components
            components = list(nx.weakly_connected_components(G))
            if not components:
                return None
                
            # Sort by size
            self.available_trees = sorted(components, key=len, reverse=True)[:5]
            
            if not self.available_trees:
                return None
                
            # Get current tree
            if self.current_tree_index >= len(self.available_trees):
                self.current_tree_index = 0
                
            current_tree = list(self.available_trees[self.current_tree_index])
            
            # Find root nodes
            root_candidates = []
            for node in current_tree:
                is_root = True
                for track in self.lineage_tracks:
                    if 'children' in track and node in track['children']:
                        if track['ID'] in current_tree:
                            is_root = False
                            break
                if is_root:
                    root_candidates.append(node)
            
            # Return first root or smallest ID
            return root_candidates[0] if root_candidates else min(current_tree)
    
    def update_ui_for_current_tree(self):
        """Update UI for the current tree"""
        # Update tree counter
        if self.cell_radio.isChecked():
            self.tree_counter.setText("Custom Tree")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
        else:
            self.tree_counter.setText(f"Tree {self.current_tree_index + 1}/{len(self.available_trees)}")
            self.prev_button.setEnabled(len(self.available_trees) > 1)
            self.next_button.setEnabled(len(self.available_trees) > 1)
    
    def next_tree(self):
        """Show next tree"""
        if not self.available_trees or len(self.available_trees) <= 1:
            return
            
        self.current_tree_index = (self.current_tree_index + 1) % len(self.available_trees)
        self.generate_visualization()
    
    def previous_tree(self):
        """Show previous tree"""
        if not self.available_trees or len(self.available_trees) <= 1:
            return
            
        self.current_tree_index = (self.current_tree_index - 1) % len(self.available_trees)
        self.generate_visualization()
    
    def save_visualization(self):
        """Save the tree visualization as an image"""
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Lineage Tree", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
            
        if output_path:
            self.figure.savefig(output_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Success", f"Image saved to {output_path}")
    
    def show_time_comparison(self):
        """Open the time comparison dialog"""
        pub.sendMessage("show_time_comparison_request", lineage_tracks=self.lineage_tracks)