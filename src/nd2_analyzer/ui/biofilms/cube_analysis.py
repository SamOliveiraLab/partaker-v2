from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSlider, QSpinBox, QComboBox,
                               QGroupBox, QListWidget, QProgressBar, 
                               QCheckBox, QFrame, QTextEdit, QSplitter,
                               QFileDialog, QAbstractItemView)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QFont
import os
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage import measure
import tifffile
import traceback


class AnalysisWorker(QObject):
    """Worker thread for cube analysis to keep UI responsive"""
    
    # Signals to communicate with main thread
    progress_updated = Signal(int, str)  # progress percentage, status message
    colony_completed = Signal(str, dict)  # colony name, results
    analysis_finished = Signal(dict)  # all results
    analysis_error = Signal(str)  # error message
    console_output = Signal(str)  # console messages
    
    def __init__(self, colony_data, cube_size, selected_params):
        super().__init__()
        self.colony_data = colony_data  # List of (colony_name, colony_folder) tuples
        self.cube_size = cube_size
        self.selected_params = selected_params
        self.should_stop = False
    
    def stop_analysis(self):
        """Stop the analysis"""
        self.should_stop = True
        self.console_output.emit("Analysis cancellation requested...")
    
    def run_analysis(self):
        """Main analysis function running in background thread"""
        try:
            self.console_output.emit("=== Starting Cube Analysis ===")
            self.console_output.emit(f"Processing {len(self.colony_data)} colonies")
            self.console_output.emit(f"Square size: {self.cube_size} pixels")
            self.console_output.emit(f"Selected parameters: {self.selected_params}")
            
            all_results = {}
            
            # Calculate total files for progress
            total_files = 0
            for colony_name, colony_folder in self.colony_data:
                if os.path.exists(colony_folder):
                    tiff_files = [f for f in os.listdir(colony_folder) if f.endswith('.tiff')]
                    total_files += len(tiff_files)
            
            self.console_output.emit(f"Total files to process: {total_files}")
            processed_files = 0
            
            # Process each colony
            for colony_idx, (colony_name, colony_folder) in enumerate(self.colony_data):
                if self.should_stop:
                    self.console_output.emit("Analysis stopped by user")
                    return
                
                if os.path.exists(colony_folder):
                    self.console_output.emit(f"\n--- Processing {colony_name} ---")
                    self.progress_updated.emit(
                        int((processed_files / total_files) * 100),
                        f"Processing {colony_name}..."
                    )
                    
                    colony_results, files_processed = self.process_colony_time_series(
                        colony_folder, processed_files, total_files
                    )
                    
                    all_results[colony_name] = colony_results
                    processed_files += files_processed
                    
                    self.colony_completed.emit(colony_name, colony_results)
                    self.console_output.emit(f"Completed {colony_name}: {files_processed} files processed")
            
            self.console_output.emit(f"\n=== Analysis Complete ===")
            self.console_output.emit(f"Total files processed: {processed_files}")
            self.analysis_finished.emit(all_results)
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}\n{traceback.format_exc()}"
            self.console_output.emit(f"ERROR: {error_msg}")
            self.analysis_error.emit(str(e))
    
    def process_colony_time_series(self, colony_folder, files_processed_so_far, total_files):
        """Process all time points for a single colony"""
        colony_results = {}
        files_processed = 0
        
        # Get all TIFF files
        tiff_files = sorted([f for f in os.listdir(colony_folder) if f.endswith('.tiff')])
        self.console_output.emit(f"  Found {len(tiff_files)} time points")
        
        for i, tiff_file in enumerate(tiff_files):
            if self.should_stop:
                break
                
            file_path = os.path.join(colony_folder, tiff_file)
            
            try:
                # Update progress
                current_progress = int(((files_processed_so_far + i) / total_files) * 100)
                self.progress_updated.emit(current_progress, f"Processing {tiff_file}...")
                
                if i % 5 == 0:  # Console output every 5th file
                    self.console_output.emit(f"    Processing {tiff_file} ({i+1}/{len(tiff_files)})")
                
                # Read and process image
                image_data = tifffile.imread(file_path)
                
                # Simple approach: treat entire cropped image as biofilm
                colony_mask_and_boxes = self.extract_cropped_colony_region(image_data)
                
                # Calculate parameters
                time_point_results = self.calculate_square_parameters(
                    image_data, colony_mask_and_boxes, self.cube_size
                )
                
                # Store results
                time_point = tiff_file.split('_')[0].replace('T', '')
                colony_results[time_point] = time_point_results
                files_processed += 1
                
                # Log square count for first few files
                if i < 3:
                    self.console_output.emit(f"      -> {len(time_point_results['square_positions'])} squares analyzed")
                
            except Exception as e:
                self.console_output.emit(f"    ERROR processing {tiff_file}: {str(e)}")
                files_processed += 1
                continue
        
        return colony_results, files_processed
    
    def extract_colony_region(self, image_data):
        """Extract colony region from exported image using black-bordered box detection"""
        # Convert to grayscale if RGB
        if len(image_data.shape) == 3:
            gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_data
        
        # Try to find black-bordered boxes first
        detected_boxes = self.find_black_bordered_boxes(gray_image)
        
        if detected_boxes:
            self.console_output.emit(f"      Found {len(detected_boxes)} black-bordered boxes")
            
            # Create combined mask from all detected boxes
            combined_mask = np.zeros_like(gray_image, dtype=np.uint8)
            
            for i, (x, y, w, h) in enumerate(detected_boxes):
                self.console_output.emit(f"      Box {i+1}: x={x}, y={y}, w={w}, h={h}")
                
                # Validate box bounds
                if (x >= 0 and y >= 0 and 
                    x + w <= gray_image.shape[1] and 
                    y + h <= gray_image.shape[0]):
                    
                    # Extract region within the box
                    box_region = gray_image[y:y+h, x:x+w]
                    
                    # Validate box region
                    if box_region.size > 0:
                        # Apply thresholding within the box region
                        if np.std(box_region) > 0:  # Avoid division by zero
                            threshold_value = np.mean(box_region) + np.std(box_region) * 0.5
                            box_mask = (box_region > threshold_value).astype(np.uint8)
                        else:
                            # Fallback for uniform regions
                            box_mask = (box_region > np.mean(box_region)).astype(np.uint8)
                        
                        self.console_output.emit(f"      Box {i+1} mask: {np.sum(box_mask)} pixels detected")
                        
                        # Place the mask back into the combined mask
                        combined_mask[y:y+h, x:x+w] = box_mask
                    else:
                        self.console_output.emit(f"      Warning: Empty box region for box {i+1}")
                else:
                    self.console_output.emit(f"      Warning: Box {i+1} bounds invalid: x={x}, y={y}, w={w}, h={h}")
            
            return combined_mask, detected_boxes
        else:
            self.console_output.emit("      No red-bordered boxes found, using fallback method")
            
            # Fallback to original method
            threshold_value = np.mean(gray_image) + np.std(gray_image) * 0.5
            binary_mask = (gray_image > threshold_value).astype(np.uint8)
            
            self.console_output.emit(f"      Fallback threshold: {threshold_value:.1f}, mask pixels: {np.sum(binary_mask)}")
            return binary_mask, []
    
    def calculate_square_parameters(self, image_data, colony_mask_and_boxes, square_size):
        """Calculate parameters for each square in the colony region with offset support"""
        if len(image_data.shape) == 3:
            gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_data
        
        # Handle both old and new return formats
        if isinstance(colony_mask_and_boxes, tuple):
            colony_mask, detected_boxes = colony_mask_and_boxes
        else:
            colony_mask = colony_mask_and_boxes
            detected_boxes = []
        
        height, width = gray_image.shape
        results = {
            'square_positions': [],
            'local_density': [],
            'distance_to_edge': [],
            'distance_to_center': [],
            'shape_area': [],
            'intensity_mean': [],
            'local_thickness': []
        }
        
        # Calculate colony center of mass
        colony_center = self.calculate_colony_center_of_mass(colony_mask)
        
        # Create distance transform for edge calculations
        distance_transform = distance_transform_edt(colony_mask)
        
        # Use larger steps for speed
        step_size = max(square_size // 2, 5)
        
        squares_processed = 0
        
        # If we have detected boxes, focus analysis on those regions
        if detected_boxes:
            for box_idx, (box_x, box_y, box_w, box_h) in enumerate(detected_boxes):
                self.console_output.emit(f"      Analyzing box {box_idx+1}/{len(detected_boxes)}")
                
                # Validate box dimensions and ensure we have room for squares
                if box_w < square_size or box_h < square_size:
                    self.console_output.emit(f"      Skipping box {box_idx+1}: too small for {square_size}px squares")
                    continue
                
                # Calculate safe bounds for analysis
                end_y = min(box_y + box_h - square_size + 1, height - square_size + 1)
                end_x = min(box_x + box_w - square_size + 1, width - square_size + 1)
                
                if end_y <= box_y or end_x <= box_x:
                    self.console_output.emit(f"      Skipping box {box_idx+1}: insufficient space for analysis")
                    continue
                
                # Analyze within this box only
                for y in range(box_y, end_y, step_size):
                    for x in range(box_x, end_x, step_size):
                        if self.should_stop:
                            break
                            
                        square_center = (x + square_size//2, y + square_size//2)
                        
                        # Quick check - only process if square center is in biofilm
                        # Validate square center coordinates
                        if (0 <= square_center[1] < height and 
                            0 <= square_center[0] < width and
                            colony_mask[square_center[1], square_center[0]] > 0):
                            # Extract square regions
                            image_square = gray_image[y:y+square_size, x:x+square_size]
                            mask_square = colony_mask[y:y+square_size, x:x+square_size]
                            
                            # Only detailed analysis if significant biofilm present
                            if np.sum(mask_square) > square_size * square_size * 0.1:
                                # Store global coordinates
                                results['square_positions'].append(square_center)
                                
                                # Calculate parameters based on selection
                                if 'Local Density' in self.selected_params:
                                    results['local_density'].append(
                                        np.sum(mask_square) / (square_size * square_size)
                                    )
                                else:
                                    results['local_density'].append(0)
                                
                                if 'Distance to Edge' in self.selected_params:
                                    results['distance_to_edge'].append(
                                        distance_transform[square_center[1], square_center[0]]
                                    )
                                else:
                                    results['distance_to_edge'].append(0)
                                
                                if 'Distance to Center' in self.selected_params:
                                    results['distance_to_center'].append(
                                        self.calc_distance_to_center(square_center, colony_center)
                                    )
                                else:
                                    results['distance_to_center'].append(0)
                                
                                results['shape_area'].append(np.sum(mask_square))
                                
                                if 'Fluorescence Intensity' in self.selected_params:
                                    results['intensity_mean'].append(
                                        self.calc_intensity_mean(image_square, mask_square)
                                    )
                                else:
                                    results['intensity_mean'].append(0)
                                
                                if 'Local Texture' in self.selected_params:
                                    results['local_thickness'].append(
                                        results['distance_to_edge'][-1]
                                    )
                                else:
                                    results['local_thickness'].append(0)
                                
                                squares_processed += 1
        else:
            # Fallback to original full-image analysis
            for y in range(0, height - square_size + 1, step_size):
                for x in range(0, width - square_size + 1, step_size):
                    if self.should_stop:
                        break
                        
                    square_center = (x + square_size//2, y + square_size//2)
                    
                    # Quick check - only process if square center is in biofilm
                    if colony_mask[square_center[1], square_center[0]] > 0:
                        # Extract square regions
                        image_square = gray_image[y:y+square_size, x:x+square_size]
                        mask_square = colony_mask[y:y+square_size, x:x+square_size]
                        
                        # Only detailed analysis if significant biofilm present
                        if np.sum(mask_square) > square_size * square_size * 0.1:
                            results['square_positions'].append(square_center)
                            
                            # Calculate parameters based on selection
                            if 'Local Density' in self.selected_params:
                                results['local_density'].append(
                                    np.sum(mask_square) / (square_size * square_size)
                                )
                            else:
                                results['local_density'].append(0)
                            
                            if 'Distance to Edge' in self.selected_params:
                                results['distance_to_edge'].append(
                                    distance_transform[square_center[1], square_center[0]]
                                )
                            else:
                                results['distance_to_edge'].append(0)
                            
                            if 'Distance to Center' in self.selected_params:
                                results['distance_to_center'].append(
                                    self.calc_distance_to_center(square_center, colony_center)
                                )
                            else:
                                results['distance_to_center'].append(0)
                            
                            results['shape_area'].append(np.sum(mask_square))
                            
                            if 'Fluorescence Intensity' in self.selected_params:
                                results['intensity_mean'].append(
                                    self.calc_intensity_mean(image_square, mask_square)
                                )
                            else:
                                results['intensity_mean'].append(0)
                            
                            if 'Local Texture' in self.selected_params:
                                results['local_thickness'].append(
                                    results['distance_to_edge'][-1]
                                )
                            else:
                                results['local_thickness'].append(0)
                            
                            squares_processed += 1
        
        return results
    
    def calc_distance_to_center(self, square_center, colony_center):
        """Distance to colony centroid"""
        return np.sqrt((square_center[0] - colony_center[0])**2 + (square_center[1] - colony_center[1])**2)
    
    def calc_intensity_mean(self, image_square, mask_square):
        """Mean intensity of pixels within square"""
        biofilm_pixels = image_square[mask_square > 0]
        return np.mean(biofilm_pixels) if len(biofilm_pixels) > 0 else 0
    
    def calculate_colony_center_of_mass(self, colony_mask):
        """Calculate center of mass of colony"""
        y_coords, x_coords = np.where(colony_mask > 0)
        if len(x_coords) == 0:
            return (0, 0)
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        return (center_x, center_y)
    
    def extract_colony_region_direct(self, image_data):
        """Direct red pixel detection for exported colony images"""
        self.console_output.emit(f"      Processing exported image: {image_data.shape}, dtype: {image_data.dtype}")
        
        # Handle RGB images with red borders
        if len(image_data.shape) == 3:
            self.console_output.emit("      RGB image - detecting red borders")
            
            # Convert to float for calculations
            red_channel = image_data[:, :, 0].astype(np.float32)
            green_channel = image_data[:, :, 1].astype(np.float32)
            blue_channel = image_data[:, :, 2].astype(np.float32)
            
            # Detect red pixels: red significantly higher than green/blue
            red_threshold = np.percentile(red_channel, 99)  # Top 1% of red values
            red_mask = ((red_channel > green_channel * 2) & 
                       (red_channel > blue_channel * 2) & 
                       (red_channel > red_threshold * 0.8))
            
            red_pixel_count = np.sum(red_mask)
            self.console_output.emit(f"      Found {red_pixel_count} red pixels")
            
            if red_pixel_count > 100:  # Minimum red pixels for border detection
                # Find connected components of red pixels
                try:
                    labeled_regions = measure.label(red_mask)
                    regions = measure.regionprops(labeled_regions)
                    
                    detected_boxes = []
                    
                    for region in regions:
                        # Get bounding box
                        min_row, min_col, max_row, max_col = region.bbox
                        w = max_col - min_col
                        h = max_row - min_row
                        
                        # Filter for reasonable colony sizes
                        if (50 <= w <= 1000 and 50 <= h <= 1000 and region.area > 50):
                            # Add padding around detected region
                            padding = 10
                            x = max(0, min_col - padding)
                            y = max(0, min_row - padding)
                            w = min(image_data.shape[1] - x, w + 2*padding)
                            h = min(image_data.shape[0] - y, h + 2*padding)
                            
                            detected_boxes.append((x, y, w, h))
                            self.console_output.emit(f"      Red region {len(detected_boxes)}: x={x}, y={y}, w={w}, h={h}")
                    
                    if detected_boxes:
                        self.console_output.emit(f"      Found {len(detected_boxes)} red-bordered regions")
                        
                        # Convert to grayscale for biofilm analysis
                        gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
                        combined_mask = np.zeros_like(gray_image, dtype=np.uint8)
                        
                        for i, (x, y, w, h) in enumerate(detected_boxes):
                            # Extract and threshold biofilm region
                            box_region = gray_image[y:y+h, x:x+w]
                            threshold_value = np.mean(box_region) + np.std(box_region) * 0.5
                            box_mask = (box_region > threshold_value).astype(np.uint8)
                            combined_mask[y:y+h, x:x+w] = box_mask
                            
                            biofilm_pixels = np.sum(box_mask)
                            self.console_output.emit(f"      Region {i+1}: {biofilm_pixels} biofilm pixels")
                        
                        total_biofilm = np.sum(combined_mask)
                        self.console_output.emit(f"      Total: {total_biofilm} biofilm pixels in red regions")
                        return combined_mask, detected_boxes
                        
                except Exception as e:
                    self.console_output.emit(f"      Error in red detection: {str(e)}")
            
            # Convert to grayscale for fallback
            gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_data.copy()
        
        # Fallback: analyze entire image
        self.console_output.emit("      No red regions found, analyzing entire image")
        threshold_value = np.mean(gray_image) + np.std(gray_image) * 0.5
        binary_mask = (gray_image > threshold_value).astype(np.uint8)
        
        biofilm_pixels = np.sum(binary_mask)
        self.console_output.emit(f"      Fallback: {biofilm_pixels} biofilm pixels detected")
        return binary_mask, []
    
    def extract_cropped_colony_region(self, image_data):
        """Simple extraction for cropped colony exports - entire image is biofilm"""
        self.console_output.emit(f"      Processing cropped colony export: {image_data.shape}, dtype: {image_data.dtype}")
        
        # Convert to grayscale if needed
        if len(image_data.shape) == 3:
            gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_data.copy()
        
        # For cropped exports, the entire image should be analyzed
        # Apply simple thresholding to identify biofilm pixels
        threshold_value = np.mean(gray_image) + np.std(gray_image) * 0.3  # Lower threshold for biofilm
        binary_mask = (gray_image > threshold_value).astype(np.uint8)
        
        biofilm_pixels = np.sum(binary_mask)
        self.console_output.emit(f"      Cropped colony analysis: {biofilm_pixels} biofilm pixels in {gray_image.shape}")
        
        # Return the entire image area as the "detected box"
        height, width = gray_image.shape
        detected_box = [(0, 0, width, height)]  # x, y, w, h for entire image
        
        return binary_mask, detected_box
    
    def extract_outlined_colony_region(self, image_data):
        """Extract colony region from outlined export format"""
        self.console_output.emit(f"      Processing outlined export: {image_data.shape}, dtype: {image_data.dtype}")
        
        # Convert to grayscale if needed
        if len(image_data.shape) == 3:
            gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_data.copy()
        
        # For outlined exports, the biofilm is typically in center region
        # Find the main biofilm region using intensity-based detection
        
        # Use adaptive approach: find brightest connected region
        height, width = gray_image.shape
        
        # Try different thresholds to find the main biofilm
        for percentile in [90, 85, 80, 75]:
            threshold = np.percentile(gray_image, percentile)
            binary_mask = (gray_image > threshold).astype(np.uint8)
            
            # Find connected components
            from skimage.measure import label, regionprops
            labeled = label(binary_mask)
            regions = regionprops(labeled)
            
            if regions:
                # Find largest region (main biofilm)
                largest_region = max(regions, key=lambda r: r.area)
                
                # Check if region is reasonable size (not too big = background, not too small = noise)
                total_pixels = height * width
                region_ratio = largest_region.area / total_pixels
                
                if 0.01 <= region_ratio <= 0.5:  # 1-50% of image
                    # Create focused region around the biofilm
                    min_row, min_col, max_row, max_col = largest_region.bbox
                    
                    # Add padding around biofilm
                    padding = 50
                    x = max(0, min_col - padding)
                    y = max(0, min_row - padding)
                    w = min(width - x, max_col - min_col + 2*padding)
                    h = min(height - y, max_row - min_row + 2*padding)
                    
                    self.console_output.emit(f"      Found biofilm region: x={x}, y={y}, w={w}, h={h} (area={largest_region.area})")
                    
                    # Create mask for this region
                    focused_mask = np.zeros_like(gray_image, dtype=np.uint8)
                    region_image = gray_image[y:y+h, x:x+w]
                    
                    # Apply refined thresholding within biofilm region
                    region_threshold = np.mean(region_image) + np.std(region_image) * 0.3
                    region_mask = (region_image > region_threshold).astype(np.uint8)
                    focused_mask[y:y+h, x:x+w] = region_mask
                    
                    biofilm_pixels = np.sum(focused_mask)
                    self.console_output.emit(f"      Focused analysis: {biofilm_pixels} biofilm pixels in {w}x{h} region")
                    
                    return focused_mask, [(x, y, w, h)]
        
        # Fallback: analyze entire image if no good region found
        self.console_output.emit("      No focused region found, using entire image")
        threshold_value = np.mean(gray_image) + np.std(gray_image) * 0.5
        binary_mask = (gray_image > threshold_value).astype(np.uint8)
        
        biofilm_pixels = np.sum(binary_mask)
        self.console_output.emit(f"      Full image analysis: {biofilm_pixels} biofilm pixels")
        return binary_mask, []
    
    def find_black_bordered_boxes(self, gray_image):
        """Find black-bordered rectangular regions using edge detection"""
        try:
            self.console_output.emit("      Detecting red-bordered boxes from exported images...")
            
            # Validate input
            if gray_image is None or gray_image.size == 0:
                self.console_output.emit("      Error: Invalid image for box detection")
                return []
                
            # Ensure image is grayscale
            if len(gray_image.shape) != 2:
                self.console_output.emit("      Error: Expected grayscale image for box detection")
                return []
            
            # Debug: Check image properties before processing
            self.console_output.emit(f"      Image dtype: {gray_image.dtype}, shape: {gray_image.shape}")
            self.console_output.emit(f"      Image range: {gray_image.min():.2f} to {gray_image.max():.2f}")
            
            # Convert to 8-bit format required by OpenCV Canny
            if gray_image.dtype != np.uint8:
                self.console_output.emit(f"      Converting from {gray_image.dtype} to uint8...")
                
                try:
                    if gray_image.dtype in [np.uint16, np.int16]:
                        # For 16-bit images, scale down to 8-bit
                        gray_8bit = (gray_image / 256).astype(np.uint8)
                        self.console_output.emit("      Used 16-bit scaling")
                    elif gray_image.dtype in [np.float32, np.float64]:
                        # For float images, normalize to 0-1 then scale to 0-255
                        gray_normalized = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min())
                        gray_8bit = (gray_normalized * 255).astype(np.uint8)
                        self.console_output.emit("      Used float normalization")
                    else:
                        # For other types, use OpenCV's normalize function
                        gray_8bit = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        self.console_output.emit(f"      Used OpenCV normalize for {gray_image.dtype}")
                        
                    self.console_output.emit(f"      Converted: dtype={gray_8bit.dtype}, range={gray_8bit.min()}-{gray_8bit.max()}")
                    
                except Exception as conv_error:
                    self.console_output.emit(f"      Conversion error: {str(conv_error)}")
                    raise conv_error
            else:
                self.console_output.emit("      Image already uint8")
                gray_8bit = gray_image
            
            try:
                # Apply Gaussian blur to reduce noise
                self.console_output.emit("      Applying Gaussian blur...")
                blurred = cv2.GaussianBlur(gray_8bit, (3, 3), 0)
                self.console_output.emit(f"      Blur successful: {blurred.dtype}")
                
                # Apply Canny edge detection with conservative thresholds
                self.console_output.emit("      Applying Canny edge detection...")
                edges = cv2.Canny(blurred, 50, 150)
                self.console_output.emit(f"      Canny successful: {edges.shape}")
                
            except Exception as processing_error:
                self.console_output.emit(f"      Processing error: {str(processing_error)}")
                self.console_output.emit(f"      Image info at error: dtype={gray_8bit.dtype if 'gray_8bit' in locals() else 'unknown'}")
                raise processing_error
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.console_output.emit("      No contours found")
                return []
            
            detected_boxes = []
            
            for contour in contours:
                try:
                    # Check contour size first
                    if cv2.contourArea(contour) < 400:  # Minimum area threshold
                        continue
                        
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if the approximated contour has 4 points (rectangular)
                    if len(approx) == 4:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Filter by size constraints (20-300 pixels for width/height)
                        if (20 <= w <= 800 and 20 <= h <= 800 and 
                            x > 0 and y > 0 and 
                            x + w < gray_image.shape[1] and y + h < gray_image.shape[0]):
                            
                            # Check if the border is actually dark (black)
                            try:
                                border_region = gray_image[y:y+h, x:x+w]
                                
                                # Validate border region
                                if border_region.size == 0:
                                    continue
                                
                                # Sample border pixels (top, bottom, left, right edges)
                                top_border = border_region[0, :]
                                bottom_border = border_region[-1, :]
                                left_border = border_region[:, 0]
                                right_border = border_region[:, -1]
                                
                                # Combine all border pixels
                                border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])
                                
                                # Check if border is significantly darker than image mean
                                border_mean = np.mean(border_pixels)
                                image_mean = np.mean(gray_image)
                                
                                # Red borders appear bright in grayscale conversion  
                                # Look for borders that are brighter than average (red shows as bright)
                                if border_mean > image_mean * 1.1:  # Border is brighter than average
                                    detected_boxes.append((x, y, w, h))
                                    self.console_output.emit(f"      Detected box: x={x}, y={y}, w={w}, h={h}, border_mean={border_mean:.1f}")
                                    
                            except Exception as border_error:
                                self.console_output.emit(f"      Error checking border for contour: {str(border_error)}")
                                continue
                                
                except Exception as contour_error:
                    self.console_output.emit(f"      Error processing contour: {str(contour_error)}")
                    continue
            
            # Sort boxes by area (largest first) and limit to reasonable number
            detected_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
            detected_boxes = detected_boxes[:10]  # Limit to top 10 boxes
            
            self.console_output.emit(f"      Final detection: {len(detected_boxes)} valid boxes")
            return detected_boxes
            
        except Exception as e:
            self.console_output.emit(f"      Error in box detection: {str(e)}")
            import traceback
            self.console_output.emit(f"      Traceback: {traceback.format_exc()}")
            return []


class CubeAnalysisWidget(QWidget):
    """Widget for cube-based analysis of exported colony time series"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State variables
        self.selected_colonies = []
        self.cube_size = 10
        self.analysis_results = {}
        self.base_folder = ""
        
        # Threading
        self.worker_thread = None
        self.worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Configuration
        left_widget = self.create_configuration_panel()
        main_splitter.addWidget(left_widget)
        
        # Right side - Results
        right_widget = self.create_results_panel()
        main_splitter.addWidget(right_widget)
        
        # Set splitter proportions (30% left, 70% right)
        main_splitter.setSizes([300, 700])
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(main_splitter)
    
    def create_configuration_panel(self):
        """Create the left configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title_label = QLabel("Cube Analysis Configuration")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2196F3; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Input Selection Group
        input_group = self.create_input_selection_group()
        layout.addWidget(input_group)
        
        # Cube Configuration Group
        cube_config_group = self.create_cube_config_group()
        layout.addWidget(cube_config_group)
        
        # Processing Controls Group
        processing_group = self.create_processing_group()
        layout.addWidget(processing_group)
        
        # Stretch to push everything to top
        layout.addStretch()
        
        return widget
    
    def create_input_selection_group(self):
        """Create input selection group"""
        group = QGroupBox("1. Select Colony Data")
        layout = QVBoxLayout(group)
        
        # Browse folder button
        browse_layout = QHBoxLayout()
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setStyleSheet("color: #666; font-style: italic;")
        browse_btn = QPushButton("Browse Export Folder")
        browse_btn.clicked.connect(self.browse_export_folder)
        browse_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        browse_layout.addWidget(self.folder_path_label)
        browse_layout.addWidget(browse_btn)
        layout.addLayout(browse_layout)
        
        # Colony list
        layout.addWidget(QLabel("Available Colonies:"))
        self.colony_list = QListWidget()
        self.colony_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.colony_list.setMaximumHeight(150)
        layout.addWidget(self.colony_list)
        
        # Quick selection buttons
        selection_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_colonies)
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_no_colonies)
        
        selection_layout.addWidget(select_all_btn)
        selection_layout.addWidget(select_none_btn)
        layout.addLayout(selection_layout)
        
        return group
    
    def create_cube_config_group(self):
        """Create cube configuration group"""
        group = QGroupBox("2. Cube Parameters")
        layout = QVBoxLayout(group)
        
        # Cube size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Square Size:"))
        
        self.cube_size_slider = QSlider(Qt.Horizontal)
        self.cube_size_slider.setMinimum(5)
        self.cube_size_slider.setMaximum(50)
        self.cube_size_slider.setValue(10)
        self.cube_size_slider.valueChanged.connect(self.on_cube_size_changed)
        
        self.cube_size_label = QLabel("10 pixels")
        self.cube_size_label.setMinimumWidth(60)
        
        size_layout.addWidget(self.cube_size_slider)
        size_layout.addWidget(self.cube_size_label)
        layout.addLayout(size_layout)
        
        # Parameter selection
        layout.addWidget(QLabel("Parameters to Calculate:"))
        
        # Essential parameters
        self.density_check = QCheckBox("Local Density")
        self.density_check.setChecked(True)
        self.edge_distance_check = QCheckBox("Distance to Edge")
        self.edge_distance_check.setChecked(True)
        self.center_distance_check = QCheckBox("Distance to Center")
        self.center_distance_check.setChecked(True)
        self.texture_check = QCheckBox("Local Texture")
        self.texture_check.setChecked(True)
        
        # Advanced parameters
        self.fluorescence_check = QCheckBox("Fluorescence Intensity")
        self.roughness_check = QCheckBox("Surface Roughness")
        
        for checkbox in [self.density_check, self.edge_distance_check, 
                        self.center_distance_check, self.texture_check,
                        self.fluorescence_check, self.roughness_check]:
            layout.addWidget(checkbox)
        
        return group
    
    def create_processing_group(self):
        """Create processing controls group"""
        group = QGroupBox("3. Processing")
        layout = QVBoxLayout(group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        # Start analysis button
        self.start_analysis_btn = QPushButton("Start Cube Analysis")
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        self.start_analysis_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.start_analysis_btn.setEnabled(False)
        
        # Cancel button
        self.cancel_analysis_btn = QPushButton("Cancel")
        self.cancel_analysis_btn.clicked.connect(self.cancel_analysis)
        self.cancel_analysis_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.cancel_analysis_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_analysis_btn)
        button_layout.addWidget(self.cancel_analysis_btn)
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Select colony data to begin")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        return group
    
    def create_results_panel(self):
        """Create the right results panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Results title
        results_title = QLabel("Analysis Results")
        results_font = QFont()
        results_font.setBold(True)
        results_font.setPointSize(14)
        results_title.setFont(results_font)
        results_title.setStyleSheet("color: #2196F3; margin-bottom: 10px;")
        layout.addWidget(results_title)
        
        # Visualization controls
        viz_controls = self.create_visualization_controls()
        layout.addWidget(viz_controls)
        
        # Console output area
        console_label = QLabel("Console Output:")
        console_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(console_label)
        
        self.console_area = QTextEdit()
        self.console_area.setMaximumHeight(200)
        self.console_area.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-family: 'Courier New', monospace;")
        layout.addWidget(self.console_area)
        
        # Results display area
        results_label = QLabel("Analysis Results:")
        results_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(results_label)
        
        self.results_area = QTextEdit()
        self.results_area.setPlaceholderText("Analysis results will appear here...")
        self.results_area.setMaximumHeight(200)
        layout.addWidget(self.results_area)
        
        # Export section
        export_section = self.create_export_section()
        layout.addWidget(export_section)
        
        # Stretch
        layout.addStretch()
        
        return widget
    
    def create_visualization_controls(self):
        """Create visualization controls"""
        group = QGroupBox("Visualization")
        layout = QHBoxLayout(group)
        
        layout.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems(["Local Density", "Distance to Edge", "Distance to Center", "Local Texture"])
        layout.addWidget(self.param_combo)
        
        layout.addWidget(QLabel("Time Point:"))
        self.time_combo = QComboBox()
        layout.addWidget(self.time_combo)
        
        self.visualize_btn = QPushButton("Generate Heatmap")
        self.visualize_btn.setEnabled(False)
        layout.addWidget(self.visualize_btn)
        
        layout.addStretch()
        
        return group
    
    def create_export_section(self):
        """Create export section"""
        group = QGroupBox("Export Results")
        layout = QHBoxLayout(group)
        
        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.setEnabled(False)
        self.export_plots_btn = QPushButton("Export Plots")
        self.export_plots_btn.setEnabled(False)
        
        layout.addWidget(self.export_csv_btn)
        layout.addWidget(self.export_plots_btn)
        layout.addStretch()
        
        return group
    
    # Event handlers
    def browse_export_folder(self):
        """Browse for exported colony folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Colony Export Folder")
        if folder:
            self.base_folder = folder
            self.folder_path_label.setText(os.path.basename(folder))
            self.scan_colony_folders(folder)
    
    def scan_colony_folders(self, base_folder):
        """Scan for colony folders and populate list"""
        self.colony_list.clear()
        
        # Look for Colony_XXX folders
        colony_folders = []
        for item in os.listdir(base_folder):
            if item.startswith("Colony_") and os.path.isdir(os.path.join(base_folder, item)):
                colony_folders.append(item)
        
        colony_folders.sort()
        
        for folder in colony_folders:
            # Count TIFF files in folder
            folder_path = os.path.join(base_folder, folder)
            tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]
            
            list_item = f"{folder} ({len(tiff_files)} time points)"
            self.colony_list.addItem(list_item)
        
        if colony_folders:
            self.start_analysis_btn.setEnabled(True)
            self.status_label.setText(f"Found {len(colony_folders)} colonies. Select colonies and start analysis.")
        else:
            self.status_label.setText("No colony folders found in selected directory.")
    
    def select_all_colonies(self):
        """Select all colonies"""
        for i in range(self.colony_list.count()):
            self.colony_list.item(i).setSelected(True)
    
    def select_no_colonies(self):
        """Deselect all colonies"""
        for i in range(self.colony_list.count()):
            self.colony_list.item(i).setSelected(False)
    
    def on_cube_size_changed(self, value):
        """Handle cube size slider change"""
        self.cube_size = value
        self.cube_size_label.setText(f"{value} pixels")
    
    def get_selected_parameters(self):
        """Get list of selected parameters"""
        params = []
        if self.density_check.isChecked():
            params.append("Local Density")
        if self.edge_distance_check.isChecked():
            params.append("Distance to Edge")
        if self.center_distance_check.isChecked():
            params.append("Distance to Center")
        if self.texture_check.isChecked():
            params.append("Local Texture")
        if self.fluorescence_check.isChecked():
            params.append("Fluorescence Intensity")
        if self.roughness_check.isChecked():
            params.append("Surface Roughness")
        
        return params
    
    # ============================================================================
    # THREADING AND ANALYSIS CONTROL
    # ============================================================================
    
    def start_analysis(self):
        """Start cube analysis in background thread"""
        selected_items = self.colony_list.selectedItems()
        if not selected_items:
            self.status_label.setText("Please select at least one colony.")
            return
        
        # Prepare colony data
        colony_data = []
        for item in selected_items:
            colony_name = item.text().split(' ')[0]
            colony_folder = os.path.join(self.base_folder, colony_name)
            colony_data.append((colony_name, colony_folder))
        
        # Clear console
        self.console_area.clear()
        self.results_area.clear()
        
        # Setup worker thread
        self.worker_thread = QThread()
        self.worker = AnalysisWorker(colony_data, self.cube_size, self.get_selected_parameters())
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.colony_completed.connect(self.on_colony_completed)
        self.worker.analysis_finished.connect(self.on_analysis_finished)
        self.worker.analysis_error.connect(self.on_analysis_error)
        self.worker.console_output.connect(self.on_console_output)
        
        # Connect thread signals
        self.worker_thread.started.connect(self.worker.run_analysis)
        self.worker_thread.finished.connect(self.cleanup_thread)
        
        # Update UI
        self.start_analysis_btn.setEnabled(False)
        self.cancel_analysis_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Start thread
        self.worker_thread.start()
    
    def cancel_analysis(self):
        """Cancel the current analysis"""
        if self.worker:
            self.worker.stop_analysis()
    
    def cleanup_thread(self):
        """Clean up the worker thread"""
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None
        
        # Reset UI
        self.start_analysis_btn.setEnabled(True)
        self.cancel_analysis_btn.setEnabled(False)
    
    # ============================================================================
    # SIGNAL HANDLERS
    # ============================================================================
    
    def on_progress_updated(self, progress, status):
        """Handle progress updates from worker"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
    
    def on_colony_completed(self, colony_name, results):
        """Handle completion of single colony"""
        # Log completion with basic stats from results
        if results and 'square_positions' in next(iter(results.values()), {}):
            total_squares = sum(len(tp_data.get('square_positions', [])) for tp_data in results.values())
            self.console_area.append(f"✓ Completed {colony_name} - {total_squares} total squares analyzed")
        else:
            self.console_area.append(f"✓ Completed {colony_name}")
        self.console_area.ensureCursorVisible()
    
    
    def on_analysis_finished(self, all_results):
        """Handle completion of entire analysis"""
        self.analysis_results = all_results
        self.display_results()
        
        # Populate time point dropdown
        self.populate_time_combo()
        
        # Enable buttons
        self.visualize_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)
        self.export_plots_btn.setEnabled(True)
        
        # Connect the export buttons (add these connections)
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_plots_btn.clicked.connect(self.export_plots)
        self.visualize_btn.clicked.connect(self.generate_heatmap)
        
        self.status_label.setText("Analysis complete!")
        self.progress_bar.setValue(100)
        
        # Cleanup
        self.cleanup_thread()

    def populate_time_combo(self):
        """Populate time point dropdown with available time points"""
        self.time_combo.clear()
        
        if self.analysis_results:
            # Get time points from first colony
            first_colony = list(self.analysis_results.values())[0]
            time_points = sorted(first_colony.keys(), key=lambda x: int(x))
            
            for tp in time_points:
                self.time_combo.addItem(f"T{tp.zfill(3)}")

    def export_to_csv(self):
        """Export analysis results to CSV file"""
        if not self.analysis_results:
            self.status_label.setText("No results to export.")
            return
        
        try:
            # Get save location
            from PySide6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV File", "biofilm_cube_analysis.csv", "CSV Files (*.csv)"
            )
            
            if not file_path:
                return
            
            import csv
            
            # Create CSV with all data
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                header = ['Colony', 'TimePoint', 'Square_X', 'Square_Y', 'Local_Density', 
                        'Distance_To_Edge', 'Distance_To_Center', 'Shape_Area', 
                        'Intensity_Mean', 'Local_Thickness']
                writer.writerow(header)
                
                # Write data
                for colony_name, colony_data in self.analysis_results.items():
                    for time_point, time_data in colony_data.items():
                        for i in range(len(time_data['square_positions'])):
                            x, y = time_data['square_positions'][i]
                            row = [
                                colony_name,
                                time_point,
                                x, y,
                                time_data['local_density'][i],
                                time_data['distance_to_edge'][i],
                                time_data['distance_to_center'][i],
                                time_data['shape_area'][i],
                                time_data['intensity_mean'][i],
                                time_data['local_thickness'][i]
                            ]
                            writer.writerow(row)
            
            self.status_label.setText(f"Data exported to {file_path}")
            self.console_area.append(f"✓ CSV exported: {file_path}")
            
        except Exception as e:
            self.status_label.setText(f"Export failed: {str(e)}")
            print(f"CSV export error: {e}")

    def generate_heatmap(self):
        """Generate heatmap visualization"""
        if not self.analysis_results:
            self.status_label.setText("No results to visualize.")
            return
        
        try:
            # Get selected parameter and time point
            selected_param = self.param_combo.currentText()
            selected_time_text = self.time_combo.currentText()
            
            if not selected_time_text:
                self.status_label.setText("Please select a time point.")
                return
            
            # Extract time point number
            time_point = selected_time_text.replace('T', '').lstrip('0') or '0'
            
            # Get first colony data (extend later for multiple colonies)
            first_colony = list(self.analysis_results.keys())[0]
            colony_data = self.analysis_results[first_colony]
            
            if time_point not in colony_data:
                self.status_label.setText(f"Time point {time_point} not found.")
                return
            
            time_data = colony_data[time_point]
            
            # Map parameter names to data keys
            param_map = {
                'Local Density': 'local_density',
                'Distance to Edge': 'distance_to_edge',
                'Distance to Center': 'distance_to_center',
                'Local Texture': 'local_thickness'
            }
            
            if selected_param not in param_map:
                self.status_label.setText("Parameter not available.")
                return
            
            data_key = param_map[selected_param]
            
            # Create simple heatmap
            import matplotlib.pyplot as plt
            import numpy as np
            
            positions = time_data['square_positions']
            values = time_data[data_key]
            
            # Filter out zero values if they're just placeholders
            filtered_data = [(pos, val) for pos, val in zip(positions, values) if val > 0]
            
            if not filtered_data:
                self.status_label.setText("No data to visualize for this parameter.")
                return
            
            positions, values = zip(*filtered_data)
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Create plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(x_coords, y_coords, c=values, cmap='viridis', s=20)
            plt.colorbar(scatter, label=selected_param)
            plt.title(f'{selected_param} - {first_colony} - {selected_time_text}')
            plt.xlabel('X Position (pixels)')
            plt.ylabel('Y Position (pixels)')
            plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
            
            plt.tight_layout()
            plt.show()
            
            self.status_label.setText(f"Generated heatmap for {selected_param}")
            
        except Exception as e:
            self.status_label.setText(f"Visualization failed: {str(e)}")
            print(f"Heatmap error: {e}")

    def export_plots(self):
        """Export plots to files"""
        if not self.analysis_results:
            self.status_label.setText("No results to export.")
            return
        
        try:
            # Get save directory
            from PySide6.QtWidgets import QFileDialog
            save_dir = QFileDialog.getExistingDirectory(self, "Select Directory for Plots")
            
            if not save_dir:
                return
            
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Parameters to plot
            param_map = {
                'Local Density': 'local_density',
                'Distance to Edge': 'distance_to_edge',
                'Distance to Center': 'distance_to_center',
                'Local Texture': 'local_thickness'
            }
            
            plots_created = 0
            
            for colony_name, colony_data in self.analysis_results.items():
                # Get a few representative time points
                time_points = sorted(colony_data.keys(), key=lambda x: int(x))
                sample_times = time_points[::len(time_points)//4] if len(time_points) > 4 else time_points
                
                for param_name, data_key in param_map.items():
                    for time_point in sample_times[:3]:  # Max 3 time points per parameter
                        time_data = colony_data[time_point]
                        
                        positions = time_data['square_positions']
                        values = time_data[data_key]
                        
                        # Filter non-zero values
                        filtered_data = [(pos, val) for pos, val in zip(positions, values) if val > 0]
                        
                        if filtered_data:
                            positions, values = zip(*filtered_data)
                            x_coords = [pos[0] for pos in positions]
                            y_coords = [pos[1] for pos in positions]
                            
                            # Create plot
                            plt.figure(figsize=(10, 8))
                            scatter = plt.scatter(x_coords, y_coords, c=values, cmap='viridis', s=20)
                            plt.colorbar(scatter, label=param_name)
                            plt.title(f'{param_name} - {colony_name} - T{time_point.zfill(3)}')
                            plt.xlabel('X Position (pixels)')
                            plt.ylabel('Y Position (pixels)')
                            plt.gca().invert_yaxis()
                            
                            # Save plot
                            filename = f"{colony_name}_{param_name.replace(' ', '_')}_T{time_point.zfill(3)}.png"
                            filepath = os.path.join(save_dir, filename)
                            plt.savefig(filepath, dpi=300, bbox_inches='tight')
                            plt.close()
                            
                            plots_created += 1
            
            self.status_label.setText(f"Exported {plots_created} plots to {save_dir}")
            self.console_area.append(f"✓ Exported {plots_created} plots")
            
        except Exception as e:
            self.status_label.setText(f"Plot export failed: {str(e)}")
            print(f"Plot export error: {e}")
    
    
    def on_analysis_error(self, error_msg):
        """Handle analysis errors"""
        self.status_label.setText(f"Analysis failed: {error_msg}")
        self.console_area.append(f"ERROR: {error_msg}")
        self.cleanup_thread()
    
    def on_console_output(self, message):
        """Handle console output from worker"""
        self.console_area.append(message)
        self.console_area.ensureCursorVisible()
    
    def display_results(self):
        """Display analysis results in the text area"""
        if not self.analysis_results:
            return
        
        results_text = "Analysis Results Summary:\n\n"
        
        total_squares = 0
        total_timepoints = 0
        
        for colony_name, colony_data in self.analysis_results.items():
            results_text += f"{colony_name}:\n"
            results_text += f"  Time points analyzed: {len(colony_data)}\n"
            total_timepoints += len(colony_data)
            
            # Calculate average squares per timepoint
            squares_per_timepoint = []
            for timepoint_data in colony_data.values():
                squares_count = len(timepoint_data['square_positions'])
                squares_per_timepoint.append(squares_count)
                total_squares += squares_count
            
            if squares_per_timepoint:
                avg_squares = np.mean(squares_per_timepoint)
                results_text += f"  Average squares per timepoint: {avg_squares:.1f}\n"
                
                # Show sample parameter values from first timepoint
                first_timepoint = list(colony_data.keys())[0]
                first_data = colony_data[first_timepoint]
                
                if first_data['local_density'] and any(d > 0 for d in first_data['local_density']):
                    avg_density = np.mean([d for d in first_data['local_density'] if d > 0])
                    results_text += f"  Sample local density: {avg_density:.3f}\n"
                
                if first_data['distance_to_edge'] and any(d > 0 for d in first_data['distance_to_edge']):
                    avg_edge_dist = np.mean([d for d in first_data['distance_to_edge'] if d > 0])
                    results_text += f"  Sample distance to edge: {avg_edge_dist:.1f} pixels\n"
            
            results_text += "\n"
        
        results_text += f"Total Analysis Summary:\n"
        results_text += f"  Total squares analyzed: {total_squares}\n"
        results_text += f"  Total time points: {total_timepoints}\n"
        if len(self.analysis_results) > 0:
            results_text += f"  Average squares per colony: {total_squares / len(self.analysis_results):.1f}\n"
        
        self.results_area.setText(results_text)