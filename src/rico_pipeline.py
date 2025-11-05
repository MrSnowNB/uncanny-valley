"""
RICo Pipeline for Phase 2

End-to-end pipeline for real-time mouth synchronization.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import time

try:
    # Try relative imports (when run as module)
    from .mouth_tracker import MouthROITracker
    from .phoneme_aligner import PhonemeAligner
    from .viseme_mapper import VisemeMapper
    from .roi_compositor import ROICompositor
    from .video_duration_matcher import VideoDurationMatcher
except ImportError:
    # Fallback to absolute imports (when run directly)
    from mouth_tracker import MouthROITracker
    from phoneme_aligner import PhonemeAligner
    from viseme_mapper import VisemeMapper
    from roi_compositor import ROICompositor
    from video_duration_matcher import VideoDurationMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RicoPipeline:
    """End-to-end RICo Phase 2 pipeline"""

    def __init__(self):
        """Initialize all RICo components"""
        self.mouth_tracker = MouthROITracker()
        self.phoneme_aligner = PhonemeAligner()
        self.viseme_mapper = VisemeMapper()
        self.roi_compositor = ROICompositor()
        self.video_matcher = VideoDurationMatcher()

        logger.info("🚀 RICo Pipeline initialized")

    def process_video_with_audio(self,
                               video_path: str,
                               audio_path: str,
                               text: str,
                               output_path: Optional[str] = None) -> str:
        """
        Process video with synchronized mouth movements

        Args:
            video_path: Path to source video
            audio_path: Path to TTS audio
            text: Transcript text
            output_path: Optional output path

        Returns:
            Path to processed video
        """
        if output_path is None:
            output_path = f"outputs/video/rico_{Path(video_path).stem}_{int(time.time())}.mp4"

        # Step 1: Extract phonemes from audio
        logger.info("🎤 Extracting phonemes from audio...")
        phonemes = self.phoneme_aligner.align_phonemes(text, audio_path)

        # Step 2: Map phonemes to visemes
        logger.info("🎭 Mapping phonemes to visemes...")
        visemes = self.viseme_mapper.map_phonemes_to_visemes(phonemes)
        visemes = self.viseme_mapper.apply_coarticulation(visemes)

        # Step 3: Process video frames
        logger.info("🎬 Processing video frames...")
        processed_frames = self._process_video_frames(video_path, visemes)

        # Step 4: Create output video
        logger.info("💾 Creating output video...")
        self._create_output_video(processed_frames, video_path, output_path)

        logger.info(f"✅ RICo processing complete: {output_path}")
        return output_path

    def _process_video_frames(self, video_path: str, visemes: List[Dict]) -> List[np.ndarray]:
        """
        Process video frames with mouth synchronization

        Args:
            video_path: Source video path
            visemes: Viseme timeline

        Returns:
            List of processed frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        logger.info(f"📹 Processing {frame_count} frames at {fps} FPS (duration: {duration:.2f}s)")

        processed_frames = []
        current_viseme_idx = 0
        frame_time = 0

        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Get current viseme
            current_viseme = self._get_viseme_at_time(visemes, frame_time, current_viseme_idx)
            if current_viseme:
                current_viseme_idx = current_viseme['index']

                # Track mouth in current frame
                roi_data, confidence = self.mouth_tracker.extract_mouth_roi(frame)

                if roi_data and confidence > 0.5:
                    # Apply viseme morphing
                    viseme_params = {
                        'mouth_open': current_viseme['mouth_open'],
                        'lip_round': current_viseme['lip_round']
                    }

                    # Composite morphed mouth back onto frame
                    frame = self.roi_compositor.composite_mouth_roi(
                        frame,
                        roi_data['roi'],
                        roi_data['bbox'],
                        viseme_params
                    )

            processed_frames.append(frame)
            frame_time += 1.0 / fps

        cap.release()
        logger.info(f"✅ Processed {len(processed_frames)} frames")
        return processed_frames

    def _get_viseme_at_time(self, visemes: List[Dict], time: float, start_idx: int = 0) -> Optional[Dict]:
        """
        Get viseme active at given time

        Args:
            visemes: Viseme timeline
            time: Current time in seconds
            start_idx: Starting search index

        Returns:
            Active viseme dict with index, or None
        """
        for i in range(start_idx, len(visemes)):
            viseme = visemes[i]
            viseme_end = viseme['time'] + viseme['duration']

            if viseme['time'] <= time < viseme_end:
                return {**viseme, 'index': i}

        return None

    def _create_output_video(self, frames: List[np.ndarray], source_video: str, output_path: str):
        """
        Create output video from processed frames

        Args:
            frames: List of processed frames
            source_video: Source video path (for properties)
            output_path: Output video path
        """
        if not frames:
            raise ValueError("No frames to write")

        # Get video properties from source
        cap = cv2.VideoCapture(source_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        logger.info(f"💾 Video saved: {output_path}")

# Test function
if __name__ == "__main__":
    pipeline = RicoPipeline()

    # Test with existing files
    audio_files = list(Path("outputs/audio").glob("*.wav"))
    if audio_files:
        audio_path = str(audio_files[0])
        video_path = "data/video_clips/speaking-neutral.mp4"
        text = "This is a test of the RICo pipeline."

        try:
            output = pipeline.process_video_with_audio(
                video_path, audio_path, text
            )
            print(f"✅ RICo pipeline test completed: {output}")
        except Exception as e:
            print(f"❌ RICo pipeline test failed: {e}")
    else:
        print("❌ No audio files found for testing")
