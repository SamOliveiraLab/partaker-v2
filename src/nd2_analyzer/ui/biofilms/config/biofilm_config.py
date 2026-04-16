import json
import os
from typing import Dict, Any

class BiofilmConfig:
    """Configuration management for biofilm analysis specific settings"""
    
    BIOFILM_CONFIG_FILE = "biofilm_analysis_config.json"
    
    def __init__(self):
        # Default biofilm analysis parameters
        self.default_settings = {
            # Colony grouping parameters
            "colony_grouping": {
                "connection_distance": 10,  # pixels
                "min_colony_size": 100,     # minimum area in pixels
                "max_gap_size": 5,          # maximum gap within colony
            },
            
            # Tracking parameters
            "colony_tracking": {
                "max_displacement": 50,     # maximum pixels moved between frames
                "overlap_threshold": 0.3,   # minimum overlap for tracking
                "track_min_length": 3,      # minimum frames for valid track
            },
            
            # Spatial analysis
            "spatial_analysis": {
                "grid_size": 20,           # size of analysis grid squares
                "smoothing_radius": 15,    # radius for spatial smoothing
                "edge_detection_sigma": 2, # edge detection parameter
            },
            
            # Visualization
            "visualization": {
                "default_colormap": "viridis",
                "show_colony_ids": True,
                "show_tracks": True,
                "track_color_by": "motility",  # or "age", "size"
            },
            
            # Cloud dynamics parameters
            "cloud_dynamics": {
                "movement_threshold": 2.0,    # minimum movement to consider significant
                "growth_threshold": 1.1,      # minimum area ratio for growth
                "division_min_fragments": 2,  # minimum pieces for division event
            }
        }
        
        self.current_settings = self.default_settings.copy()
        self.load_config()
    
    def load_config(self):
        """Load biofilm configuration from file"""
        if os.path.exists(self.BIOFILM_CONFIG_FILE):
            try:
                with open(self.BIOFILM_CONFIG_FILE, 'r') as f:
                    saved_settings = json.load(f)
                    
                # Merge saved settings with defaults (in case new parameters were added)
                self._merge_settings(self.current_settings, saved_settings)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading biofilm config: {e}. Using defaults.")
    
    def _merge_settings(self, default_dict: Dict[str, Any], saved_dict: Dict[str, Any]):
        """Recursively merge saved settings with defaults"""
        for key, value in saved_dict.items():
            if key in default_dict:
                if isinstance(value, dict) and isinstance(default_dict[key], dict):
                    self._merge_settings(default_dict[key], value)
                else:
                    default_dict[key] = value
    
    def save_config(self):
        """Save biofilm configuration to file"""
        try:
            with open(self.BIOFILM_CONFIG_FILE, 'w') as f:
                json.dump(self.current_settings, f, indent=2)
        except Exception as e:
            print(f"Error saving biofilm config: {e}")
    
    def get_setting(self, category: str, parameter: str, default=None):
        """Get a specific setting value"""
        return self.current_settings.get(category, {}).get(parameter, default)
    
    def set_setting(self, category: str, parameter: str, value: Any, save: bool = True):
        """Set a specific setting value"""
        if category not in self.current_settings:
            self.current_settings[category] = {}
        
        self.current_settings[category][parameter] = value
        
        if save:
            self.save_config()
    
    def get_colony_grouping_params(self) -> Dict[str, Any]:
        """Get colony grouping parameters"""
        return self.current_settings["colony_grouping"]
    
    def get_tracking_params(self) -> Dict[str, Any]:
        """Get colony tracking parameters"""
        return self.current_settings["colony_tracking"]
    
    def get_spatial_params(self) -> Dict[str, Any]:
        """Get spatial analysis parameters"""
        return self.current_settings["spatial_analysis"]
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.current_settings = self.default_settings.copy()
        self.save_config()