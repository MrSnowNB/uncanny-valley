"""
ROI Compositor for RICo Phase 2

Composites mouth ROIs onto video frames using seamless blending.
"""

import cv2
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ROICompositor:
    """Composites mouth regions onto video frames"""

    def __init__(self):
        logger.info("🎨 ROICompositor initialized")

    def composite_mouth_roi(self,
                           target_frame: np.ndarray,
                           mouth_roi: np.ndarray,
                           roi_bbox: tuple,
                           viseme_params: Optional[Dict] = None) -> np.ndarray:
        """
        Composite mouth ROI onto target frame

        Args:
            target_frame: Target video frame (BGR)
            mouth_roi: Mouth region to composite (BGR)
            roi_bbox: (x, y, w, h) bounding box in target frame
            viseme_params: Optional viseme parameters for morphing

        Returns:
            Composited frame
        """
        if mouth_roi is None or mouth_roi.size == 0:
            return target_frame

        x, y, w, h = roi_bbox

        # Ensure ROI fits within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, target_frame.shape[1] - x)
        h = min(h, target_frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return target_frame

        # Resize mouth ROI to match target dimensions
        resized_roi = cv2.resize(mouth_roi, (w, h))

        # Create mask for seamless cloning
        # Use the mouth region as mask (assuming it's already segmented)
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Apply viseme morphing if parameters provided
        if viseme_params:
            resized_roi = self._apply_viseme_morphing(resized_roi, viseme_params)

        try:
            # Use seamless cloning for natural blending
            center = (x + w // 2, y + h // 2)

            # Clone the mouth region onto the target frame
            result = cv2.seamlessClone(
                resized_roi,           # Source (mouth)
                target_frame,          # Destination (face)
                mask,                  # Mask
                center,                # Center point
                cv2.NORMAL_CLONE       # Clone method
            )

            logger.debug(f"✅ Composited mouth ROI at ({x}, {y}) {w}x{h}")
            return result

        except Exception as e:
            logger.warning(f"❌ Seamless clone failed: {e}, using direct overlay")
            # Fallback: direct overlay
            result = target_frame.copy()
            result[y:y+h, x:x+w] = resized_roi
            return result

    def _apply_viseme_morphing(self, mouth_roi: np.ndarray, viseme_params: Dict) -> np.ndarray:
        """
        Apply viseme-based morphing to mouth ROI

        Args:
            mouth_roi: Original mouth region
            viseme_params: Viseme parameters (mouth_open, lip_round, etc.)

        Returns:
            Morphed mouth region
        """
        # This is a simplified morphing implementation
        # In a full implementation, this would use facial landmarks
        # to morph the mouth shape according to viseme parameters

        mouth_open = viseme_params.get('mouth_open', 0.5)
        lip_round = viseme_params.get('lip_round', 0.0)

        # Simple vertical scaling based on mouth openness
        h, w = mouth_roi.shape[:2]
        new_h = int(h * (0.5 + mouth_open * 0.5))  # Scale 0.5x to 1.0x

        if new_h != h:
            scaled = cv2.resize(mouth_roi, (w, new_h))
            # Pad or crop to original size
            if new_h > h:
                # Crop from center
                start_y = (new_h - h) // 2
                morphed = scaled[start_y:start_y + h, :]
            else:
                # Pad with black
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                start_y = (h - new_h) // 2
                padded[start_y:start_y + new_h, :] = scaled
                morphed = padded
        else:
            morphed = mouth_roi.copy()

        # Simple color adjustment for lip rounding effect
        if lip_round > 0.3:
            # Add slight reddish tint for rounded lips
            morphed = cv2.addWeighted(morphed, 0.8,
                                    np.full_like(morphed, [50, 20, 20]), 0.2, 0)

        return morphed

    def create_mouth_mask(self, mouth_roi: np.ndarray) -> np.ndarray:
        """
        Create binary mask for mouth region

        Args:
            mouth_roi: Mouth region image

        Returns:
            Binary mask (255 for mouth pixels, 0 for background)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)

        # Simple thresholding to create mask
        # In a real implementation, this would use more sophisticated segmentation
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

# Test function
if __name__ == "__main__":
    compositor = ROICompositor()

    # Create test frames
    target_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    target_frame[:, :] = [200, 150, 100]  # Brown background

    # Create test mouth ROI
    mouth_roi = np.zeros((100, 150, 3), dtype=np.uint8)
    mouth_roi[:, :] = [0, 0, 255]  # Red mouth

    # Test compositing
    roi_bbox = (245, 200, 150, 100)  # Center of frame
    result = compositor.composite_mouth_roi(target_frame, mouth_roi, roi_bbox)

    print("✅ ROI compositing test completed")
    print(f"   Target frame size: {target_frame.shape}")
    print(f"   Mouth ROI size: {mouth_roi.shape}")
    print(f"   Result size: {result.shape}")

    # Save test result
    cv2.imwrite("outputs/debug/test_composite.jpg", result)
    print("✅ Test result saved to outputs/debug/test_composite.jpg")
