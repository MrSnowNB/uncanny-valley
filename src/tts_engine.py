#!/usr/bin/env python3
"""
Alice in Cyberland - TTS Engine Module
Uses pyttsx3 for offline speech synthesis compatible with Python 3.13
"""

import os
import pyttsx3
import yaml
from pathlib import Path
import uuid

class AliceTTSEngine:
    """TTS engine using pyttsx3 for Alice's voice synthesis"""

    def __init__(self, manifest_path="data/video_manifest.yaml"):
        """Initialize TTS engine with video manifest configuration"""
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()
        self.voice_settings = self.manifest.get('voice_settings', {})
        self.engine = None
        self._initialize_engine()
        if self.engine is None:
            raise ValueError("Failed to initialize TTS engine")

    def _load_manifest(self):
        """Load video manifest YAML configuration"""
        try:
            with open(self.manifest_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading manifest: {e}")
            return {}

    def _initialize_engine(self):
        """Initialize pyttsx3 engine with Alice's voice characteristics"""
        self.engine = pyttsx3.init()
        if self.engine is None:
            print("Error: Failed to initialize TTS engine")
            return

        # Configure voice settings from manifest
        rate = self.voice_settings.get('sample_rate', 22050)
        self.engine.setProperty('rate', int(rate))  # Default speech rate

        # Select female voice if available
        self._select_female_voice()

        print("✅ TTS engine initialized with pyttsx3")

    def _select_female_voice(self):
        """Select female voice from available voices"""
        if self.engine is None:
            return
        voices = self.engine.getProperty('voices')
        if voices:
            # Find female voice
            for voice in voices:  # type: ignore
                if 'female' in voice.name.lower() or 'woman' in voice.name.lower() or 'girl' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    print(f"Selected voice: {voice.name}")
                    return

            # Fallback to first female or just use default
            for voice in voices:  # type: ignore
                if voice.name and 'voice' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    print(f"Fallback voice: {voice.name}")
                    return

        print("Using default voice")

    def synthesize(self, text, output_dir="outputs/audio", filename=None):
        """
        Generate speech audio from text

        Args:
            text: Text to synthesize
            output_dir: Directory to save audio file
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to generated audio file
        """
        if self.engine is None:
            print("Error: TTS engine not initialized")
            return None

        # Create output directory if needed
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate unique filename if not provided
        if filename is None:
            filename = f"alice_response_{uuid.uuid4().hex[:8]}.wav"

        audio_path = os.path.join(output_dir, filename)

        try:
            # Save to file directly
            self.engine.save_to_file(text, audio_path)
            self.engine.runAndWait()
            print(f"✅ Audio generated: {audio_path}")
            return audio_path
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None

    def get_audio_duration(self, audio_path):
        """
        Estimate audio duration for synchronization
        Basic implementation - returns estimated seconds
        """
        # pyttsx3 doesn't provide direct duration, so estimate
        # Rough estimate: 150 words per minute = 2.5 words per second
        word_count = len(audio_path.split('_')[-1].split('.')[0])  # rough text length
        return max(2, word_count / 2.5)  # minimum 2 seconds

    def list_voice_properties(self):
        """Debug: List current voice properties"""
        if self.engine is None:
            print("TTS engine not initialized")
            return
        voices = self.engine.getProperty('voices')
        for voice in voices:  # type: ignore
            print(f"Voice: {voice.name}, ID: {voice.id}, Age: {voice.age}")

        rate = self.engine.getProperty('rate')
        print(f"Current rate: {rate}")

    def test_engine(self, test_text="Hello, I am Alice. Welcome to Cyberland."):
        """Test TTS engine with sample text"""
        print(f"Testing TTS with: '{test_text}'")
        audio_path = self.synthesize(test_text, output_dir="outputs/test")
        if audio_path:
            duration = self.get_audio_duration(audio_path)
            print(".1f")
            return audio_path
        return None


# Test if run directly
if __name__ == "__main__":
    engine = AliceTTSEngine()
    engine.list_voice_properties()
    test_path = engine.test_engine()
    if test_path:
        print(f"Test audio saved to: {test_path}")
