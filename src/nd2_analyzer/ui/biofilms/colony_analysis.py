import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops
from typing import List, Dict, Tuple
import polars as pl

class ColonyGrouper:
    """Groups individual cells into biofilm colonies based on spatial proximity"""
    
    def __init__(self, connection_distance=10, min_colony_size=5):
        """
        Args:
            connection_distance: Maximum distance between cells to be considered connected
            min_colony_size: Minimum number of cells to form a colony
        """
        self.connection_distance = connection_distance
        self.min_colony_size = min_colony_size
    
    def group_cells_into_colonies(self, cell_data: pl.DataFrame) -> List[Dict]:
        """
        Group cells into colonies based on spatial proximity.
        
        Args:
            cell_data: Polars DataFrame with cell data including centroid_x, centroid_y
            
        Returns:
            List of colony dictionaries with colony properties
        """
        if cell_data.is_empty():
            return []
        
        # Convert to numpy arrays for processing
        cell_positions = cell_data.select(['centroid_x', 'centroid_y']).to_numpy()
        cell_ids = cell_data['cell_id'].to_list()
        
        # Calculate distance matrix between all cells
        distance_matrix = cdist(cell_positions, cell_positions)
        
        # Create adjacency matrix (cells are connected if distance < threshold)
        adjacency_matrix = distance_matrix < self.connection_distance
        
        # Find connected components (colonies)
        colonies = self._find_connected_components(adjacency_matrix)
        
        # Create colony data
        colony_list = []
        colony_id = 1
        
        for colony_cell_indices in colonies:
            if len(colony_cell_indices) >= self.min_colony_size:
                colony_cells = cell_data[colony_cell_indices]
                colony_properties = self._calculate_colony_properties(colony_cells, colony_id)
                colony_list.append(colony_properties)
                colony_id += 1
        
        return colony_list
    
    def _find_connected_components(self, adjacency_matrix):
        """Find connected components in adjacency matrix"""
        n_cells = adjacency_matrix.shape[0]
        visited = [False] * n_cells
        colonies = []
        
        for i in range(n_cells):
            if not visited[i]:
                colony = []
                self._dfs(adjacency_matrix, i, visited, colony)
                colonies.append(colony)
        
        return colonies
    
    def _dfs(self, adjacency_matrix, node, visited, colony):
        """Depth-first search to find connected cells"""
        visited[node] = True
        colony.append(node)
        
        for neighbor in range(len(adjacency_matrix[node])):
            if adjacency_matrix[node][neighbor] and not visited[neighbor]:
                self._dfs(adjacency_matrix, neighbor, visited, colony)
    
    def _calculate_colony_properties(self, colony_cells: pl.DataFrame, colony_id: int) -> Dict:
        """Calculate properties for a single colony"""
        # Get basic info
        position = colony_cells['position'][0]
        time = colony_cells['time'][0]
        channel = colony_cells['channel'][0]
        
        # Calculate colony properties
        colony_area = colony_cells['area'].sum()
        colony_perimeter = colony_cells['perimeter'].sum()
        
        # Calculate colony centroid (average of cell centroids weighted by area)
        total_area = colony_cells['area'].sum()
        colony_centroid_x = (colony_cells['centroid_x'] * colony_cells['area']).sum() / total_area
        colony_centroid_y = (colony_cells['centroid_y'] * colony_cells['area']).sum() / total_area
        
        # Calculate colony bounding box
        min_x = colony_cells['centroid_x'].min() - 10  # Add some padding
        max_x = colony_cells['centroid_x'].max() + 10
        min_y = colony_cells['centroid_y'].min() - 10
        max_y = colony_cells['centroid_y'].max() + 10
        
        # Calculate colony shape properties
        colony_width = max_x - min_x
        colony_height = max_y - min_y
        colony_aspect_ratio = max(colony_width, colony_height) / min(colony_width, colony_height)
        
        # Calculate colony compactness (how circular the colony is)
        colony_compactness = (4 * np.pi * colony_area) / (colony_perimeter**2) if colony_perimeter > 0 else 0
        
        return {
            "data_type": "colony",
            "position": position,
            "time": time,
            "channel": channel,
            "colony_id": colony_id,
            "cell_count": len(colony_cells),
            "colony_area": colony_area,
            "colony_perimeter": colony_perimeter,
            "colony_centroid_x": colony_centroid_x,
            "colony_centroid_y": colony_centroid_y,
            "colony_width": colony_width,
            "colony_height": colony_height,
            "colony_aspect_ratio": colony_aspect_ratio,
            "colony_compactness": colony_compactness,
            "bbox_x1": min_x,
            "bbox_y1": min_y,
            "bbox_x2": max_x,
            "bbox_y2": max_y
        }

class ColonyTracker:
    """Tracks colonies across time points to measure dynamics"""
    
    def __init__(self, max_displacement=50, overlap_threshold=0.3):
        """
        Args:
            max_displacement: Maximum distance a colony can move between frames
            overlap_threshold: Minimum overlap to consider colonies the same
        """
        self.max_displacement = max_displacement
        self.overlap_threshold = overlap_threshold
    
    def track_colonies_over_time(self, colony_data_by_time: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Track colonies across multiple time points.
        
        Args:
            colony_data_by_time: Dictionary mapping time -> list of colony data
            
        Returns:
            List of colony tracks with dynamics information
        """
        if not colony_data_by_time:
            return []
        
        time_points = sorted(colony_data_by_time.keys())
        tracks = []
        next_track_id = 1
        
        # Initialize tracks with first time point
        for colony in colony_data_by_time[time_points[0]]:
            track = {
                "track_id": next_track_id,
                "colony_data": [colony],
                "start_time": time_points[0],
                "end_time": time_points[0]
            }
            tracks.append(track)
            next_track_id += 1
        
        # Match colonies across subsequent time points
        for t in range(1, len(time_points)):
            current_time = time_points[t]
            current_colonies = colony_data_by_time[current_time]
            
            # Try to match each current colony to existing tracks
            matched_tracks = set()
            
            for colony in current_colonies:
                best_match = None
                best_score = 0
                
                for i, track in enumerate(tracks):
                    if i in matched_tracks:
                        continue
                    
                    # Get the last colony in this track
                    last_colony = track["colony_data"][-1]
                    
                    # Calculate matching score based on position and size similarity
                    score = self._calculate_match_score(last_colony, colony)
                    
                    if score > best_score and score > self.overlap_threshold:
                        best_score = score
                        best_match = i
                
                if best_match is not None:
                    # Extend existing track
                    tracks[best_match]["colony_data"].append(colony)
                    tracks[best_match]["end_time"] = current_time
                    matched_tracks.add(best_match)
                else:
                    # Create new track
                    new_track = {
                        "track_id": next_track_id,
                        "colony_data": [colony],
                        "start_time": current_time,
                        "end_time": current_time
                    }
                    tracks.append(new_track)
                    next_track_id += 1
        
        # Calculate dynamics for each track
        for track in tracks:
            track["dynamics"] = self._calculate_track_dynamics(track["colony_data"])
        
        return tracks
    
    def _calculate_match_score(self, colony1: Dict, colony2: Dict) -> float:
        """Calculate similarity score between two colonies"""
        # Distance between centroids
        dx = colony1["colony_centroid_x"] - colony2["colony_centroid_x"]
        dy = colony1["colony_centroid_y"] - colony2["colony_centroid_y"]
        distance = np.sqrt(dx**2 + dy**2)
        
        # If too far apart, no match
        if distance > self.max_displacement:
            return 0.0
        
        # Size similarity
        area1 = colony1["colony_area"]
        area2 = colony2["colony_area"]
        size_ratio = min(area1, area2) / max(area1, area2)
        
        # Combine distance and size into score
        distance_score = 1.0 - (distance / self.max_displacement)
        combined_score = (distance_score + size_ratio) / 2.0
        
        return combined_score
    
    def _calculate_track_dynamics(self, colony_sequence: List[Dict]) -> Dict:
        """Calculate dynamics properties for a colony track"""
        if len(colony_sequence) < 2:
            return {
                "track_length": len(colony_sequence),
                "total_displacement": 0,
                "average_speed": 0,
                "growth_rate": 0,
                "area_change": 0
            }
        
        first_colony = colony_sequence[0]
        last_colony = colony_sequence[-1]
        
        # Calculate displacement
        dx = last_colony["colony_centroid_x"] - first_colony["colony_centroid_x"]
        dy = last_colony["colony_centroid_y"] - first_colony["colony_centroid_y"]
        total_displacement = np.sqrt(dx**2 + dy**2)
        
        # Calculate average speed
        time_span = last_colony["time"] - first_colony["time"]
        average_speed = total_displacement / time_span if time_span > 0 else 0
        
        # Calculate growth
        area_change = last_colony["colony_area"] - first_colony["colony_area"]
        growth_rate = area_change / first_colony["colony_area"] if first_colony["colony_area"] > 0 else 0
        
        return {
            "track_length": len(colony_sequence),
            "total_displacement": total_displacement,
            "average_speed": average_speed,
            "growth_rate": growth_rate,
            "area_change": area_change,
            "start_area": first_colony["colony_area"],
            "end_area": last_colony["colony_area"]
        }