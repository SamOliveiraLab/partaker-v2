import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tifffile

class ColonyTimeSeriesExporter:
    """Export individual colonies as separate time series"""
    
    def __init__(self, colony_separator, image_data):
        self.colony_separator = colony_separator
        self.image_data = image_data
        self.export_stats = {}
        
    def export_all_colonies(self, output_folder: str, time_range: Tuple[int, int], 
                           position: int, channel: int, export_format: str = "outlined",
                           padding: int = 10, progress_callback=None) -> Dict:
        """
        Export all detected colonies as individual time series
        
        Args:
            output_folder: Base folder for export
            time_range: (start_time, end_time) tuple
            position: Position index
            channel: Channel index
            export_format: "outlined", "cropped", "masked", or "padded"
            padding: Extra pixels around colony (for padded format)
            progress_callback: Function to call with progress updates
        """
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all colonies
        colonies = self.colony_separator.get_all_colonies()
        if not colonies:
            return {"error": "No colonies detected"}
        
        # Calculate total operations for progress
        start_time, end_time = time_range
        total_operations = len(colonies) * (end_time - start_time + 1)
        current_operation = 0
        
        export_summary = {
            "total_colonies": len(colonies),
            "time_range": time_range,
            "position": position,
            "channel": channel,
            "export_format": export_format,
            "colonies_exported": []
        }
        
        # Export each colony
        for colony in colonies:
            try:
                colony_result = self._export_single_colony(
                    colony, output_folder, time_range, position, channel,
                    export_format, padding, progress_callback, current_operation, total_operations
                )
                
                export_summary["colonies_exported"].append(colony_result)
                current_operation += (end_time - start_time + 1)
                
            except Exception as e:
                print(f"Error exporting colony {colony['colony_id']}: {e}")
                continue
        
        # Save export summary
        summary_path = os.path.join(output_folder, "export_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(export_summary, f, indent=2)
        
        return export_summary
    
    
    def _export_single_colony(self, colony: Dict, output_folder: str, 
                         time_range: Tuple[int, int], position: int, channel: int,
                         export_format: str, padding: int, progress_callback,
                         base_progress: int, total_operations: int) -> Dict:
        
        """Export a single colony across all time points"""
        
        colony_id = colony['colony_id']
        colony_folder = os.path.join(output_folder, f"Colony_{colony_id:03d}")
        os.makedirs(colony_folder, exist_ok=True)
        
        # Get colony polygon or contour
        if 'polygon_points' in colony:
            polygon = np.array(colony['polygon_points'], dtype=np.int32)
        elif 'contour' in colony:
            polygon = colony['contour'].reshape(-1, 2).astype(np.int32)
        else:
            # Fallback to bounding box
            x1, y1, x2, y2 = colony['bbox']
            polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        
        # Calculate bounding box
        bbox = cv2.boundingRect(polygon)
        x, y, w, h = bbox
        
        start_time, end_time = time_range
        exported_files = []
        
        # Export each time point
        for t in range(start_time, end_time + 1):
            try:
                # Get raw image for this time point
                raw_image = self._get_image_frame(t, position, channel)
                print(f"Raw image shape: {raw_image.shape}")  # ADD THIS LINE

                if raw_image is None:
                    continue

                # Extract colony region based on format
                if export_format == "outlined":
                    # Full image with simple ROI outline
                    colony_region = raw_image.copy()
                    print(f"Colony region shape after copy: {colony_region.shape}")  # ADD THIS LINE
                    
                    # Convert to RGB if grayscale (needed for colored rectangle)
                    if len(colony_region.shape) == 2:
                        colony_region = cv2.cvtColor(colony_region, cv2.COLOR_GRAY2RGB)
                        print(f"Colony region shape after RGB conversion: {colony_region.shape}")  # ADD THIS LINE
                    
                    # Draw simple red rectangle around the ROI
                    cv2.rectangle(colony_region, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red box only
                    
                elif export_format == "cropped":
                    # Cropped to bounding box only
                    img_h, img_w = raw_image.shape[:2]
                    x_crop = max(0, min(x, img_w - 1))
                    y_crop = max(0, min(y, img_h - 1))
                    x2_crop = max(0, min(x + w, img_w))
                    y2_crop = max(0, min(y + h, img_h))
                    
                    colony_region = raw_image[y_crop:y2_crop, x_crop:x2_crop]
                    
                elif export_format == "padded":
                    # Cropped with padding
                    img_h, img_w = raw_image.shape[:2]
                    x_padded = max(0, x - padding)
                    y_padded = max(0, y - padding)
                    x2_padded = min(img_w, x + w + padding)
                    y2_padded = min(img_h, y + h + padding)
                    
                    colony_region = raw_image[y_padded:y2_padded, x_padded:x2_padded]
                    
                elif export_format == "masked":
                    # Full image with non-colony pixels blacked out
                    mask = self._create_colony_mask(raw_image.shape[:2], polygon)
                    colony_region = raw_image.copy()
                    colony_region[mask == 0] = 0
                
                # Save the region
                filename = f"T{t:03d}_Colony{colony_id:03d}.tiff"
                filepath = os.path.join(colony_folder, filename)
                
                # Use tifffile for better TIFF support
                tifffile.imwrite(filepath, colony_region)
                exported_files.append(filename)
                
                # Update progress
                if progress_callback:
                    current_progress = base_progress + (t - start_time + 1)
                    progress_percent = int((current_progress / total_operations) * 100)
                    progress_callback(progress_percent)
                
            except Exception as e:
                print(f"Error exporting colony {colony_id} at time {t}: {e}")
                continue
        
        # Save colony metadata
        metadata = {
            "colony_id": colony_id,
            "original_polygon": polygon.tolist(),
            "bounding_box": bbox,
            "export_format": export_format,
            "time_range": time_range,
            "files_exported": len(exported_files),
            "file_list": exported_files
        }
        
        metadata_path = os.path.join(colony_folder, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

    
    def _get_image_frame(self, time: int, position: int, channel: int) -> Optional[np.ndarray]:
        """Get raw image frame from ImageData"""
        try:
            if hasattr(self.image_data, 'data'):
                # Handle different data shapes
                if len(self.image_data.data.shape) == 5:  # T, P, C, Y, X
                    frame = self.image_data.data[time, position, channel]
                elif len(self.image_data.data.shape) == 4:  # T, P, Y, X
                    frame = self.image_data.data[time, position]
                elif len(self.image_data.data.shape) == 3:  # T, Y, X
                    frame = self.image_data.data[time]
                else:
                    return None
                
                return np.array(frame)
            else:
                return None
        except Exception as e:
            print(f"Error getting frame T={time}, P={position}, C={channel}: {e}")
            return None
    
    def _create_colony_mask(self, image_shape: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
        """Create binary mask from polygon"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        return mask
    
    def get_export_formats(self) -> List[str]:
        """Get available export format options"""
        return ["outlined", "cropped", "masked", "padded"]
    
    def estimate_export_size(self, time_range: Tuple[int, int]) -> Dict:
        """Estimate the export size and file count"""
        colonies = self.colony_separator.get_all_colonies()
        start_time, end_time = time_range
        
        total_files = len(colonies) * (end_time - start_time + 1)
        
        # Rough size estimation (this would need actual image dimensions)
        estimated_size_mb = total_files * 2.0  # Larger estimate for full images
        
        return {
            "total_colonies": len(colonies),
            "total_files": total_files,
            "estimated_size_mb": estimated_size_mb,
            "time_points": end_time - start_time + 1
        }