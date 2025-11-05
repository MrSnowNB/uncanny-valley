"""
Phoneme Alignment for RICo Phase 2

Extracts timestamped phonemes from TTS audio using phonemizer
with heuristic timing distribution.
"""

import os
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhonemeAligner:
    """Extracts phoneme timings from audio using phonemizer"""

    def __init__(self, method: str = "heuristic"):
        """
        Args:
            method: Alignment method ("heuristic" for now)
        """
        self.method = method
        logger.info(f"📊 PhonemeAligner using method: {method}")

    def align_phonemes(self, text: str, audio_path: str) -> List[Dict]:
        """
        Extract timestamped phonemes from audio

        Args:
            text: Transcript text
            audio_path: Path to audio file (.wav)

        Returns:
            List of phoneme dicts with timing
        """
        try:
            from phonemizer import phonemize
            import librosa
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            return []

        # Get phonemes using phonemizer
        phonemes_result = phonemize(
            text,
            language='en-us',
            backend='espeak',
            strip=True,
            preserve_punctuation=False
        )

        # Ensure we have a string
        if isinstance(phonemes_result, list):
            # Handle different phonemize output formats
            if phonemes_result and isinstance(phonemes_result[0], (list, tuple)):
                # Flatten nested structure
                phonemes_str = ' '.join([' '.join(item) if isinstance(item, (list, tuple)) else str(item) for item in phonemes_result])
            else:
                phonemes_str = ' '.join(str(item) for item in phonemes_result)
        else:
            phonemes_str = str(phonemes_result)

        # Clean up phonemes (remove stress markers, etc.)
        phonemes_str = phonemes_str.replace('ː', '').replace('ˈ', '').replace('ˌ', '')

        # Split into individual phonemes
        phoneme_list = list(phonemes_str.replace(' ', ''))

        # Get audio duration
        try:
            audio_duration = librosa.get_duration(filename=audio_path)
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}, using 3.0s default")
            audio_duration = 3.0

        # Distribute phonemes evenly across duration
        if len(phoneme_list) == 0:
            logger.warning("No phonemes extracted from text")
            return []

        avg_duration = audio_duration / len(phoneme_list)

        result = []
        current_time = 0.0

        for phoneme in phoneme_list:
            result.append({
                'time': current_time,
                'phoneme': phoneme,
                'duration': avg_duration,
                'word': None  # Not available in heuristic mode
            })
            current_time += avg_duration

        logger.info(f"📊 Aligned {len(result)} phonemes over {audio_duration:.2f}s")
        return result

# Test function
if __name__ == "__main__":
    aligner = PhonemeAligner()

    # Test with sample text
    test_text = "Hello, this is a test of the phoneme alignment system."
    test_audio = "outputs/audio/test_alignment.wav"

    # Create test audio if it doesn't exist
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(test_text, test_audio)
        engine.runAndWait()
        print(f"Created test audio: {test_audio}")
    except Exception as e:
        print(f"Could not create test audio: {e}")
        # Use existing audio if available
        audio_files = list(Path("outputs/audio").glob("*.wav"))
        if audio_files:
            test_audio = str(audio_files[0])
            test_text = "Sample text for testing"
            print(f"Using existing audio: {test_audio}")
        else:
            print("No audio files available for testing")
            exit(1)

    # Test alignment
    phonemes = aligner.align_phonemes(test_text, test_audio)

    print(f"✅ Aligned {len(phonemes)} phonemes")
    print("First 10 phonemes:")
    for i, p in enumerate(phonemes[:10]):
        print(".3f")

    print(f"Total duration: {sum(p['duration'] for p in phonemes):.2f}s")
