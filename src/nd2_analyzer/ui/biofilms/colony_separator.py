import numpy as np
import cv2
from skimage import measure, morphology
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from typing import List, Dict, Tuple
import polars as pl

from PySide6.QtWidgets import QDialog


class ColonySeparator:
    """BiofilmQ-style colony separation tool for automatic biofilm region detection"""

    def __init__(self):
        # Manual threshold parameters (BiofilmQ style)
        self.intensity_threshold = 160
        self.kernel_size = 20

        # Otsu thresholding parameters (from notebook)
        # self.closing_radius = 1  # radius for binary closing
        # self.min_object_size = 6  # minimum size to keep (removes noise)

        # Common parameters
        self.min_colony_size = 50
        self.max_colony_size = 100000

        # Storage
        self.detected_colonies = []
        self.manual_additions = []
        self.manual_removals = []

    def detect_colonies_from_raw_image(self, raw_image: np.ndarray) -> List[Dict]:
        """
        Detect colony regions from raw microscopy image (BiofilmQ approach)

        Args:
            raw_image: Raw grayscale microscopy image

        Returns:
            List of colony dictionaries with boundaries and properties
        """
        # Normalize image to 0-1 range for thresholding
        if raw_image.max() > 1:
            normalized_image = raw_image.astype(np.float32) / raw_image.max()
        else:
            normalized_image = raw_image.astype(np.float32)

        # Apply intensity threshold (BiofilmQ step 1)
        binary_mask = (normalized_image > self.intensity_threshold).astype(np.uint8)

        # Remove small holes within biofilms
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Fill holes
        binary_mask = morphology.remove_small_holes(binary_mask.astype(bool), area_threshold=100).astype(np.uint8)

        # Remove small objects (noise) - BiofilmQ step 2
        cleaned_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=self.min_colony_size).astype(
            np.uint8)

        # Remove objects touching borders (incomplete biofilms)
        cleared_mask = clear_border(cleaned_mask).astype(np.uint8)

        # Find contours of biofilm regions
        contours, _ = cv2.findContours(cleared_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        colonies = []
        colony_id = 1

        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)

            # Filter by size
            if self.min_colony_size <= area <= self.max_colony_size:
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    centroid_x = M["m10"] / M["m00"]
                    centroid_y = M["m01"] / M["m00"]
                else:
                    centroid_x, centroid_y = x + w / 2, y + h / 2

                # Calculate additional properties
                perimeter = cv2.arcLength(contour, True)

                # Fit ellipse if contour has enough points
                if len(contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        (center_x, center_y), (major_axis, minor_axis), angle = ellipse
                        eccentricity = np.sqrt(1 - (min(major_axis, minor_axis) / max(major_axis, minor_axis)) ** 2)
                    except:
                        major_axis = minor_axis = np.sqrt(area / np.pi) * 2
                        eccentricity = 0
                        angle = 0
                else:
                    major_axis = minor_axis = np.sqrt(area / np.pi) * 2
                    eccentricity = 0
                    angle = 0

                # Calculate solidity (area / convex hull area)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Create colony data
                colony_data = {
                    'colony_id': colony_id,
                    'area': area,
                    'centroid': (centroid_x, centroid_y),
                    'bbox': (x, y, x + w, y + h),  # (x1, y1, x2, y2)
                    'bbox_width': w,
                    'bbox_height': h,
                    'perimeter': perimeter,
                    'major_axis_length': major_axis,
                    'minor_axis_length': minor_axis,
                    'eccentricity': eccentricity,
                    'orientation': angle,
                    'solidity': solidity,
                    'contour': contour.squeeze(),  # Store contour points
                    'source': 'automatic'
                }

                colonies.append(colony_data)
                colony_id += 1

        self.detected_colonies = colonies
        return colonies

    def detect_colonies_otsu(self, raw_image: np.ndarray) -> List[Dict]:
        """
        Detect colonies using threshold + contour analysis
        """

        img = raw_image

        # normalize to uint8 if needed
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Smooths image using Gaussian blur before thresholding
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)

        # threshold
        #_, thresh = cv2.threshold(img_blur, self.intensity_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, _ = cv2.threshold(img_blur,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        adjusted = ret + (self.intensity_threshold - 50)
        adjusted = np.clip(adjusted, 0, 255)

        _, thresh = cv2.threshold(img_blur, adjusted,255, cv2.THRESH_BINARY)

        # morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        thresh_grouped = thresh
        # find contours
        contours, _ = cv2.findContours(
            thresh_grouped,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = self.min_colony_size
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

        colonies = []
        colony_id = 1

        for contour in contours:
            area = cv2.contourArea(contour)

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2
                cy = y + h / 2

            # Perimeter
            perimeter = cv2.arcLength(contour, True)

            # Ellipse fit (only if possible)
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (_, _), (major_axis, minor_axis), angle = ellipse
            else:
                # fallback approximation
                major_axis = minor_axis = np.sqrt(area / np.pi) * 2
                angle = 0

            # Solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            colony_data = {
                "colony_id": colony_id,
                "area": area,
                "centroid": (cx, cy),
                "bbox": (x, y, x + w, y + h),
                "bbox_width": w,
                "bbox_height": h,
                "perimeter": perimeter,
                "major_axis_length": major_axis,
                "minor_axis_length": minor_axis,
                "orientation": angle,
                "solidity": solidity,
                "contour": contour.reshape(-1, 2)
            }

            colonies.append(colony_data)
            colony_id += 1

        self.detected_colonies = colonies
        return colonies

    def update_parameters(self,
                          intensity_threshold: float = None,
                          closing_radius: int = None,
                          min_object_size: int = None,
                          min_colony_size: int = None,
                          kernel_size: int = None,
                          max_colony_size: int = None):
        """Update detection parameters"""
        if intensity_threshold is not None:
            self.intensity_threshold = intensity_threshold
        if kernel_size is not None:
            self.kernel_size = kernel_size
        # if closing_radius is not None:
        #    self.closing_radius = closing_radius
        # if min_object_size is not None:
        #    self.min_object_size = min_object_size
        if min_colony_size is not None:
            self.min_colony_size = min_colony_size
        if max_colony_size is not None:
            self.max_colony_size = max_colony_size

        print(f"Updated parameters: intensity_threshold={self.intensity_threshold}, "
              f"min_colony_size={self.min_colony_size}")

    def add_manual_colony(self, polygon_points: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> Dict:
        """
        Add a manually drawn colony region

        Args:
            polygon_points: List of (x, y) coordinates defining the polygon
            image_shape: Shape of the image (height, width)

        Returns:
            Dictionary containing the new colony data
        """
        if len(polygon_points) < 3:
            return None

        # Create mask from polygon
        mask = np.zeros(image_shape, dtype=np.uint8)
        points_array = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)

        # Calculate properties
        region = measure.regionprops(mask)[0]

        # Calculate bounding box
        min_row, min_col, max_row, max_col = region.bbox

        colony_id = len(self.detected_colonies) + len(self.manual_additions) + 1

        colony_data = {
            'colony_id': colony_id,
            'label': colony_id,
            'area': region.area,
            'centroid': region.centroid,
            'bbox': (min_col, min_row, max_col, max_row),
            'bbox_width': max_col - min_col,
            'bbox_height': max_row - min_row,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'extent': region.extent,
            'perimeter': region.perimeter,
            'major_axis_length': region.major_axis_length,
            'minor_axis_length': region.minor_axis_length,
            'orientation': region.orientation,
            'source': 'manual',
            'polygon_points': polygon_points
        }

        self.manual_additions.append(colony_data)
        return colony_data

    def remove_colony(self, colony_id: int):
        """Remove a colony by ID"""
        # Remove from detected colonies
        self.detected_colonies = [c for c in self.detected_colonies if c['colony_id'] != colony_id]

        # Remove from manual additions
        self.manual_additions = [c for c in self.manual_additions if c['colony_id'] != colony_id]

    def get_all_colonies(self) -> List[Dict]:
        """Get all detected colonies (automatic + manual)"""
        return self.detected_colonies + self.manual_additions

    def create_colony_overlay(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create an overlay image showing detected colonies with contours (like BiofilmQ)
        """
        overlay = np.zeros((*image_shape, 3), dtype=np.uint8)

        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 192, 203),  # Pink
            (0, 128, 128),  # Teal
        ]

        all_colonies = self.get_all_colonies()

        for i, colony in enumerate(all_colonies):
            color = colors[i % len(colors)]

            # Draw contour (like BiofilmQ's red lines)
            if 'contour' in colony:
                contour = colony['contour'].reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(overlay, [contour], -1, color, 2)

                # Calculate centroid from contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                else:
                    # Fallback to contour center
                    centroid_x = int(np.mean(contour[:, 0, 0]))
                    centroid_y = int(np.mean(contour[:, 0, 1]))

            else:
                # Fallback to bounding box if no contour
                x1, y1, x2, y2 = colony['bbox']
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                # Calculate bounding box center
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)

            # Draw colony ID near centroid (FIXED POSITIONING)
            cv2.putText(overlay, f"C{colony['colony_id']}",
                        (centroid_x - 10, centroid_y), cv2.FONT_HERSHEY_DUPLEX,
                        1.8, color, 2)  # Made text bigger and bolder

            # Draw small circle at centroid
            cv2.circle(overlay, (centroid_x, centroid_y), 4, color, -1)

        return overlay

    def start_manual_selection(self):
        """Start manual colony selection using Colony ROI Selector"""
        # Get current raw image
        if not hasattr(self, 'current_raw_image') or self.current_raw_image is None:
            self.progress_label.setText("No raw image available. Load image data first.")
            return

        # Import the colony ROI selector
        from nd2_analyzer.ui.dialogs.colony_roi_selector import ColonyROISelector

        # Get existing colonies to pass to the dialog
        existing_colonies = []
        for colony in self.colony_separator.get_all_colonies():
            if 'polygon_points' in colony:
                existing_colonies.append({
                    'colony_id': colony['colony_id'],
                    'polygon': colony['polygon_points'],
                    'mask': None  # We can regenerate this if needed
                })

        # Open the colony ROI selector dialog with existing colonies
        roi_dialog = ColonyROISelector(self.current_raw_image, existing_colonies=existing_colonies, parent=self)
        roi_dialog.colonies_selected.connect(self.handle_selected_colonies)

        # Update UI state
        self.start_manual_btn.setEnabled(False)
        self.progress_label.setText("Colony ROI Selector opened. Existing colonies are preserved.")

        # Show dialog
        result = roi_dialog.exec()

        # Reset UI state
        self.start_manual_btn.setEnabled(True)

        if result == QDialog.Accepted:
            self.progress_label.setText(f"Selected {len(self.colony_separator.get_all_colonies())} colonies manually.")
        else:
            self.progress_label.setText("Colony selection cancelled.")

    def add_polygon_point(self, x: int, y: int):
        """Add a point to the current polygon being drawn"""
        self.current_polygon.append((x, y))
        print(f"Added point: ({x}, {y}). Total points: {len(self.current_polygon)}")

    def finish_current_polygon(self, image_shape: Tuple[int, int]):
        """Finish the current polygon and create a colony"""
        if len(self.current_polygon) < 3:
            print("Need at least 3 points to create a polygon")
            return None

        # Create colony from polygon
        colony_data = self.add_manual_colony(self.current_polygon, image_shape)

        # Reset current polygon
        self.current_polygon = []

        return colony_data

    def cancel_current_polygon(self):
        """Cancel the current polygon being drawn"""
        self.current_polygon = []
        print("Current polygon cancelled")

    def create_bounding_box_colony(self, x1: int, y1: int, x2: int, y2: int, image_shape: Tuple[int, int]):
        """Create a colony from a bounding box (simpler than polygon)"""
        # Create rectangle polygon from bounding box
        polygon_points = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ]

        return self.add_manual_colony(polygon_points, image_shape)

    def get_colony_at_point(self, x: int, y: int) -> Dict:
        """Get colony at the specified point (for deletion)"""
        for colony in self.get_all_colonies():
            if 'contour' in colony:
                # Check if point is inside contour
                contour = colony['contour'].reshape(-1, 1, 2).astype(np.int32)
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    return colony
            else:
                # Check if point is inside bounding box
                x1, y1, x2, y2 = colony['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return colony
        return None

    def delete_colony_at_point(self, x: int, y: int):
        """Delete colony at the specified point"""
        colony = self.get_colony_at_point(x, y)
        if colony:
            self.remove_colony(colony['colony_id'])
            print(f"Deleted colony {colony['colony_id']}")
            return True
        return False