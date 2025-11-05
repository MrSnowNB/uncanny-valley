"""
Mouth ROI Tracker for RICo Phase 2

Tracks mouth region using OpenCV Haar cascades as MediaPipe alternative
for Python 3.13 compatibility.
"""

import cv2  # type: ignore
import numpy as np
from typing import Optional, Tuple, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MouthROITracker:
    """Tracks mouth region using OpenCV face and mouth detection"""

    def __init__(self):
        """Initialize OpenCV cascade classifiers"""
        # Load Haar cascade classifiers
        # Try multiple possible paths for Haar cascades
        cv2_path = os.path.dirname(cv2.__file__)  # type: ignore
        possible_paths = [
            cv2_path.replace('__init__.py', 'data/haarcascades/'),
            '/usr/share/opencv4/haarcascades/',
            '/usr/local/share/opencv4/haarcascades/',
            os.path.join(cv2_path, 'data', 'haarcascades')
        ]

        cascade_file = None
        for path in possible_paths:
            test_file = os.path.join(path, 'haarcascade_frontalface_default.xml')
            if os.path.exists(test_file):
                cascade_file = test_file
                break

        if cascade_file:
            self.face_cascade = cv2.CascadeClassifier(cascade_file)  # type: ignore
        else:
            # Fallback: try to load from opencv-python package
            try:
                import cv2.data  # type: ignore
                self.face_cascade = cv2.CascadeClassifier(  # type: ignore
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
                )
            except:
                logger.warning("Could not load Haar cascade classifier")
                self.face_cascade = None

        # Note: OpenCV doesn't have a built-in mouth cascade
        # We'll use face detection and estimate mouth position
        # This is a simplified approach for Phase 2 proof-of-concept

        logger.info("✅ MouthROITracker initialized with OpenCV Haar cascades")

    def extract_mouth_roi(self, frame: np.ndarray) -> Tuple[Optional[Dict], float]:
        """
        Extract mouth region from frame using face detection

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            (roi_data, confidence) where roi_data contains:
                - 'roi': Cropped mouth region
                - 'polygon': Estimated mouth outline
                - 'bbox': Bounding box (x, y, w, h)
                - 'landmarks': Estimated landmarks
        """
        if self.face_cascade is None:
            logger.warning("Face cascade not loaded, cannot detect mouth ROI")
            return None, 0.0

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            logger.debug("No faces detected")
            return None, 0.0

        # Use the largest face (assuming closest/most prominent)
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face

        # Estimate mouth position (typically bottom 1/3 of face, centered)
        mouth_y = y + int(h * 0.6)  # Start of mouth region
        mouth_h = int(h * 0.25)     # Height of mouth region
        mouth_x = x + int(w * 0.25) # Left edge
        mouth_w = int(w * 0.5)      # Width

        # Ensure coordinates are within frame bounds
        mouth_x = max(0, mouth_x)
        mouth_y = max(0, mouth_y)
        mouth_w = min(mouth_w, frame.shape[1] - mouth_x)
        mouth_h = min(mouth_h, frame.shape[0] - mouth_y)

        if mouth_w <= 0 or mouth_h <= 0:
            return None, 0.0

        # Extract mouth ROI
        roi = frame[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w].copy()

        # Create estimated mouth polygon (rectangle approximation)
        mouth_polygon = np.array([
            [mouth_x, mouth_y],
            [mouth_x + mouth_w, mouth_y],
            [mouth_x + mouth_w, mouth_y + mouth_h],
            [mouth_x, mouth_y + mouth_h]
        ], dtype=np.int32)

        # Package results
        roi_data = {
            'roi': roi,
            'polygon': mouth_polygon,
            'bbox': (mouth_x, mouth_y, mouth_w, mouth_h),
            'landmarks': {
                'estimated_mouth_region': True,
                'face_bbox': (x, y, w, h)
            },
            'frame_size': (frame.shape[1], frame.shape[0])
        }

        # Confidence based on face detection confidence (simplified)
        confidence = 0.7  # Fixed confidence for Haar cascade detection

        logger.debug(f"✅ Mouth ROI extracted: {mouth_w}x{mouth_h}px at ({mouth_x}, {mouth_y})")

        return roi_data, confidence

    def visualize_mouth_roi(self, frame: np.ndarray, roi_data: Optional[Dict]) -> np.ndarray:
        """Draw mouth ROI and face detection on frame (for debugging)"""
        if roi_data is None:
            return frame

        debug_frame = frame.copy()

        # Draw face bounding box
        if 'face_bbox' in roi_data.get('landmarks', {}):
            fx, fy, fw, fh = roi_data['landmarks']['face_bbox']
            cv2.rectangle(debug_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)  # type: ignore

        # Draw mouth bounding box
        x, y, w, h = roi_data['bbox']
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # type: ignore

        # Draw mouth polygon
        cv2.polylines(debug_frame, [roi_data['polygon']], True, (0, 0, 255), 2)  # type: ignore

        return debug_frame

    def release(self):
        """Cleanup resources (OpenCV handles this automatically)"""
        pass

# Test function
if __name__ == "__main__":
    tracker = MouthROITracker()

    # Test on a video frame
    cap = cv2.VideoCapture("data/video_clips/speaking-neutral.mp4")  # type: ignore
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            roi_data, confidence = tracker.extract_mouth_roi(frame)
            if roi_data:
                print(f"✅ Mouth detected with confidence: {confidence}")
                print(f"   ROI size: {roi_data['roi'].shape}")
                print(f"   BBox: {roi_data['bbox']}")
            else:
                print("❌ No mouth detected")
        cap.release()
    else:
        print("❌ Could not open test video")

    tracker.release()
