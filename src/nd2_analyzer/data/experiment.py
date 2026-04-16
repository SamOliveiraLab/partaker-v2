import os
import json
import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional
import nd2

class Experiment:
    """
    Represents a time-lapse microscopy experiment.
    
    Attributes:
        name (str): Name of the experiment
        image_files (List[str]): List of paths to image files (ND2/TIFF)
        interval (float): Time step between frames in seconds
        rpu_values (Dict[str, float]): Dictionary of RPU values
        tiff_types (Dict[str, str]): Dictionary mapping TIFF files to their types
        nd2_files (List[str]): Legacy - List of paths to ND2 files
    """
    
    def __init__(self, name: str, image_files: List[str], interval: float, 
                 rpu_values: Dict[str, float] = None, tiff_types: Dict[str, str] = None):
        """
        Initialize an experiment.
        
        Args:
            name: Name of the experiment
            image_files: List of paths to image files (ND2 or TIFF)
            interval: Time step between frames in seconds
            rpu_values: Dictionary of RPU values (optional)
            tiff_types: Dictionary mapping TIFF files to their types (optional)
        """
        self.name = name
        self.image_files = image_files if image_files else []
        self.tiff_types = tiff_types if tiff_types else {}
        
        # Legacy support
        self.nd2_files = [f for f in self.image_files if f.lower().endswith('.nd2')]
        self.interval = interval
        self.rpu_values = rpu_values or {}
        
    def add_nd2_file(self, file_path: str) -> None:
        """
        Add a new ND2 file to the experiment.
        
        Args:
            file_path: Path to the ND2 file
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be opened as an ND2 file or if its shape is incompatible
        """
        # Check if file already exists in the list
        if file_path in self.nd2_files:
            return  # File already added, nothing to do
            
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
            
        # Check if ND2 works
        try:
            import nd2
            reader = nd2.ND2File(file_path)
        except ImportError:
            raise ImportError("The nd2 module is required. Please install it with 'pip install nd2'.")
        except Exception as e:
            raise ValueError(f"Error opening ND2 file {file_path}: {str(e)}")
            
        # Check if the shape is compatible
        new_shape = reader.shape
        if self.nd2_files:  # If we already have files
            try:
                first_file = self.nd2_files[0]
                first_reader = nd2.ND2File(first_file)
                first_shape = first_reader.shape
                
                # We assume compatibility means same shape except for the time dimension (first dimension)
                if len(new_shape) != len(first_shape):
                    reader.close()
                    first_reader.close()
                    raise ValueError(f"File {file_path} has different dimensions ({len(new_shape)}) than existing files ({len(first_shape)}).")
                    
                if new_shape[1:] != first_shape[1:]:
                    reader.close()
                    first_reader.close()
                    raise ValueError(f"File {file_path} shape {new_shape} is not compatible with existing files shape {first_shape}.")
                    
                first_reader.close()
            except Exception as e:
                reader.close()
                raise ValueError(f"Error checking compatibility: {str(e)}")
                
        # If all checks pass, add the file
        self.nd2_files.append(file_path)
        reader.close()
        
    def get_image_manager(self) -> 'ImageManager':
        """
        Create and return an ImageManager for this experiment.
        
        Returns:
            An ImageManager instance for this experiment
        """
        return ImageManager(self)
    
    def save(self, file_path: str) -> None:
        """
        Save experiment configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration
        """
        config = {
            'name': self.name,
            'nd2_files': self.nd2_files,
            'interval': self.interval,
            'rpu_values': self.rpu_values
        }
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def load(cls, file_path: str) -> 'Experiment':
        """
        Load experiment configuration from a JSON file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            An Experiment instance
        """
        with open(file_path, 'r') as f:
            config = json.load(f)
            
        return cls(
            name=config['name'],
            nd2_files=config['nd2_files'],
            interval=config['interval'],
            rpu_values=config['rpu_values']
        )
