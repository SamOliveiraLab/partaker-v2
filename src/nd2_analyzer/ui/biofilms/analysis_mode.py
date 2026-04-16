from enum import Enum
import json
import os
from typing import Optional

class AnalysisMode(Enum):
    """Analysis mode enumeration for Partaker application"""
    SINGLE_CELL = "single_cell"
    BIOFILM_CLOUD = "biofilm_cloud"

    def __str__(self):
        return self.value

    @property
    def display_name(self):
        """Human-readable display name for the mode"""
        return {
            AnalysisMode.SINGLE_CELL: "Single Cell Analysis",
            AnalysisMode.BIOFILM_CLOUD: "Biofilm Cloud Analysis"
        }[self]

    @property
    def description(self):
        """Description of what each mode does"""
        return {
            AnalysisMode.SINGLE_CELL: "Track and analyze individual bacterial cells, their morphology, lineage trees, and motility patterns.",
            AnalysisMode.BIOFILM_CLOUD: "Analyze biofilm colonies as 'clouds' - track colony movement, growth, division, and spatial dynamics over time."
        }[self]

class AnalysisModeConfig:
    """Configuration manager for analysis modes"""

    CONFIG_FILE = "partaker_config.json"

    def __init__(self):
        self.current_mode = AnalysisMode.SINGLE_CELL
        self.default_mode = AnalysisMode.SINGLE_CELL
        self.remember_choice = True
        self.load_config()

    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)

                mode_str = config_data.get('analysis_mode', 'single_cell')
                self.current_mode = AnalysisMode(mode_str)
                self.default_mode = AnalysisMode(config_data.get('default_mode', 'single_cell'))
                self.remember_choice = config_data.get('remember_choice', True)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                self.current_mode = AnalysisMode.SINGLE_CELL

    def save_config(self):
        """Save configuration to file"""
        config_data = {
            'analysis_mode': self.current_mode.value,
            'default_mode': self.default_mode.value,
            'remember_choice': self.remember_choice,
            'version': '1.0'
        }

        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def set_mode(self, mode: AnalysisMode, save: bool = True):
        """Set the current analysis mode"""
        self.current_mode = mode
        if save:
            self.save_config()

    def get_mode(self) -> AnalysisMode:
        """Get the current analysis mode"""
        return self.current_mode

    def is_single_cell_mode(self) -> bool:
        """Check if currently in single cell mode"""
        return self.current_mode == AnalysisMode.SINGLE_CELL

    def is_biofilm_mode(self) -> bool:
        """Check if currently in biofilm mode"""
        return self.current_mode == AnalysisMode.BIOFILM_CLOUD
