"""
Standalone test for MouthROITracker

Tests mouth tracking on easiest clip (speaking-nuetral.mp4)
WITHOUT any integration with chat_server.py
"""

import cv2
from src.mouth_tracker import MouthROITracker
import os


def test_mouth_tracking_isolated():
    """Test mouth tracking in complete isolation"""

    print("Starting isolated mouth tracker test...")

    tracker = MouthROITracker()

    # Use easiest clip first
    video_path = "data/video_clips/speaking-neutral.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("Available clips:")
        for f in os.listdir("data/video_clips"):
            print(f"  - {f}")
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames_processed = 0
    frames_with_mouth = 0
    confidences = []

    # Process first 30 frames
    print("Processing frames...")
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            print(f"End of video at frame {i}")
            break

        roi_data, confidence = tracker.extract_mouth_roi(frame)
        frames_processed += 1

        if roi_data is not None:
            frames_with_mouth += 1
            confidences.append(confidence)

            # Verify ROI is valid
            assert roi_data['roi'].size > 0, "Empty ROI"
            assert len(roi_data['polygon']) > 10, "Too few landmarks"

    cap.release()
    tracker.release()

    # Calculate metrics
    detection_rate = frames_with_mouth / frames_processed if frames_processed > 0 else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    print(f"\n✅ Processed {frames_processed} frames")
    print(f"✅ Detection rate: {detection_rate*100:.1f}%")
    print(f"✅ Avg confidence: {avg_confidence:.2f}")

    # Assertions
    assert frames_processed > 0, "No frames processed"
    assert detection_rate > 0.5, f"Detection rate too low: {detection_rate*100:.1f}%"
    assert avg_confidence > 0.5, f"Confidence too low: {avg_confidence:.2f}"

    print("\n✅ TEST PASSED - Mouth tracking working")
    return True


if __name__ == "__main__":
    try:
        test_mouth_tracking_isolated()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
