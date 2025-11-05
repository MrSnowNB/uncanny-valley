"""
Viseme Mapping for RICo Phase 2

Maps phonemes to visemes (visual mouth shapes) for lip-sync animation.
"""

from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisemeMapper:
    """Maps phonemes to visemes for mouth animation"""

    # Simplified IPA to viseme mapping
    # Visemes represent distinct mouth shapes
    PHONEME_TO_VISEME = {
        # Vowels - open mouth shapes
        'a': 'AA', 'ɑ': 'AA', 'æ': 'AE', 'ʌ': 'AH', 'ɔ': 'AO', 'aʊ': 'AW', 'aɪ': 'AY',
        'ɛ': 'EH', 'ɝ': 'ER', 'eɪ': 'EY', 'ɪ': 'IH', 'i': 'IY', 'oʊ': 'OW', 'ɔɪ': 'OY',
        'ʊ': 'UH', 'u': 'UW',

        # Consonants - various mouth shapes
        'b': 'MP', 'p': 'MP', 'm': 'MP',  # Bilabial
        'f': 'FV', 'v': 'FV',             # Labiodental
        'θ': 'TH', 'ð': 'TH',             # Dental
        't': 'DT', 'd': 'DT', 'n': 'DT',  # Alveolar
        's': 'SZ', 'z': 'SZ',             # Alveolar fricative
        'ʃ': 'SH', 'ʒ': 'SH',             # Postalveolar
        'tʃ': 'JH', 'dʒ': 'JH',           # Postalveolar affricate
        'k': 'KG', 'g': 'KG', 'ŋ': 'KG',  # Velar
        'h': 'HH',                        # Glottal
        'l': 'L',                         # Lateral
        'r': 'R',                         # Rhotic
        'w': 'W',                         # Labial-velar
        'j': 'Y',                         # Palatal

        # Default fallback
        'DEFAULT': 'AA'
    }

    # Viseme to mouth shape description
    VISEME_SHAPES = {
        'AA': {'mouth_open': 0.9, 'lip_round': 0.1, 'description': 'Wide open'},
        'AE': {'mouth_open': 0.8, 'lip_round': 0.0, 'description': 'Open smile'},
        'AH': {'mouth_open': 0.6, 'lip_round': 0.2, 'description': 'Neutral open'},
        'AO': {'mouth_open': 0.7, 'lip_round': 0.8, 'description': 'Round open'},
        'AW': {'mouth_open': 0.5, 'lip_round': 0.6, 'description': 'Rounded'},
        'AY': {'mouth_open': 0.4, 'lip_round': 0.3, 'description': 'Narrow'},
        'EH': {'mouth_open': 0.6, 'lip_round': 0.1, 'description': 'Mid open'},
        'ER': {'mouth_open': 0.5, 'lip_round': 0.4, 'description': 'Mid rounded'},
        'EY': {'mouth_open': 0.4, 'lip_round': 0.2, 'description': 'Narrow smile'},
        'IH': {'mouth_open': 0.3, 'lip_round': 0.1, 'description': 'Narrow open'},
        'IY': {'mouth_open': 0.2, 'lip_round': 0.1, 'description': 'Closed smile'},
        'OW': {'mouth_open': 0.4, 'lip_round': 0.7, 'description': 'Rounded narrow'},
        'OY': {'mouth_open': 0.3, 'lip_round': 0.6, 'description': 'Rounded closed'},
        'UH': {'mouth_open': 0.3, 'lip_round': 0.5, 'description': 'Mid rounded'},
        'UW': {'mouth_open': 0.2, 'lip_round': 0.8, 'description': 'Closed rounded'},

        'MP': {'mouth_open': 0.1, 'lip_round': 0.9, 'description': 'Bilabial closed'},
        'FV': {'mouth_open': 0.2, 'lip_round': 0.1, 'description': 'Teeth visible'},
        'TH': {'mouth_open': 0.3, 'lip_round': 0.0, 'description': 'Tongue visible'},
        'DT': {'mouth_open': 0.1, 'lip_round': 0.0, 'description': 'Tip of tongue'},
        'SZ': {'mouth_open': 0.2, 'lip_round': 0.0, 'description': 'Hissing'},
        'SH': {'mouth_open': 0.3, 'lip_round': 0.0, 'description': 'Retroflex'},
        'JH': {'mouth_open': 0.1, 'lip_round': 0.0, 'description': 'Affricate'},
        'KG': {'mouth_open': 0.1, 'lip_round': 0.0, 'description': 'Back of tongue'},
        'HH': {'mouth_open': 0.8, 'lip_round': 0.0, 'description': 'Breathy'},
        'L': {'mouth_open': 0.3, 'lip_round': 0.0, 'description': 'Lateral'},
        'R': {'mouth_open': 0.4, 'lip_round': 0.2, 'description': 'Rhotic'},
        'W': {'mouth_open': 0.2, 'lip_round': 0.7, 'description': 'Labial'},
        'Y': {'mouth_open': 0.2, 'lip_round': 0.1, 'description': 'Palatal'},
    }

    def __init__(self):
        logger.info("🎭 VisemeMapper initialized")

    def map_phonemes_to_visemes(self, phonemes: List[Dict]) -> List[Dict]:
        """
        Convert phoneme timeline to viseme timeline

        Args:
            phonemes: List of phoneme dicts with timing

        Returns:
            List of viseme dicts with timing and mouth shapes
        """
        visemes = []

        for phoneme in phonemes:
            viseme_name = self.PHONEME_TO_VISEME.get(
                phoneme['phoneme'],
                self.PHONEME_TO_VISEME['DEFAULT']
            )

            viseme_shape = self.VISEME_SHAPES.get(viseme_name, self.VISEME_SHAPES['AA'])

            viseme = {
                'time': phoneme['time'],
                'duration': phoneme['duration'],
                'phoneme': phoneme['phoneme'],
                'viseme': viseme_name,
                'mouth_open': viseme_shape['mouth_open'],
                'lip_round': viseme_shape['lip_round'],
                'description': viseme_shape['description']
            }

            visemes.append(viseme)

        logger.info(f"🎭 Mapped {len(phonemes)} phonemes to {len(visemes)} visemes")
        return visemes

    def apply_coarticulation(self, visemes: List[Dict], window_size: int = 2) -> List[Dict]:
        """
        Apply coarticulation smoothing to viseme sequence

        Args:
            visemes: Original viseme sequence
            window_size: Smoothing window size

        Returns:
            Smoothed viseme sequence
        """
        if len(visemes) < 3:
            return visemes

        smoothed = []

        for i, viseme in enumerate(visemes):
            # Get neighboring visemes for smoothing
            start_idx = max(0, i - window_size)
            end_idx = min(len(visemes), i + window_size + 1)
            neighbors = visemes[start_idx:end_idx]

            # Average mouth openness and lip rounding
            avg_mouth_open = sum(v['mouth_open'] for v in neighbors) / len(neighbors)
            avg_lip_round = sum(v['lip_round'] for v in neighbors) / len(neighbors)

            # Create smoothed viseme
            smoothed_viseme = viseme.copy()
            smoothed_viseme['mouth_open'] = avg_mouth_open
            smoothed_viseme['lip_round'] = avg_lip_round
            smoothed_viseme['smoothed'] = True

            smoothed.append(smoothed_viseme)

        logger.info(f"🔄 Applied coarticulation smoothing to {len(smoothed)} visemes")
        return smoothed

# Test function
if __name__ == "__main__":
    mapper = VisemeMapper()

    # Test phonemes
    test_phonemes = [
        {'time': 0.0, 'phoneme': 'h', 'duration': 0.1},
        {'time': 0.1, 'phoneme': 'ɛ', 'duration': 0.1},
        {'time': 0.2, 'phoneme': 'l', 'duration': 0.1},
        {'time': 0.3, 'phoneme': 'oʊ', 'duration': 0.2},
    ]

    # Map to visemes
    visemes = mapper.map_phonemes_to_visemes(test_phonemes)

    print("✅ Mapped phonemes to visemes:")
    for v in visemes:
        print(".2f")

    # Apply coarticulation
    smoothed = mapper.apply_coarticulation(visemes)

    print("\n✅ Applied coarticulation smoothing:")
    for v in smoothed:
        print(".2f")
