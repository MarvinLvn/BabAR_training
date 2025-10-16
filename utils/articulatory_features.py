import panphon
import torch
from typing import List, Dict
import json


class ArticulatoryFeatureExtractor:
    def __init__(self):
        self.ft = panphon.FeatureTable()
        self.feature_names = [
            'syl',      # [±syllabic]. Is the segment the nucleus of a syllable?
            'son',      # [±sonorant]. Is the segment produced with a relatively unobstructed vocal tract?
            'cons',     # [±consonantal]. Is the segment consonantal (not a vowel or glide, or laryngeal consonant)?
            'cont',     # [±continuant]. Is the segment produced with continuous oral airflow?
            'delrel',   # [±delayed release]. Is the segment an affricate?
            'lat',      # [±lateral]. Is the segment produced with a lateral constriction?
            'nas',      # [±nasal]. Is the segment produced with nasal airflow
            'strid',    # [±strident]. Is the segment produced with noisy friction?
            'voi',      # [±voice]. Are the vocal folds vibrating during the production of the segment?
            'ant',      # [±anterior]. Is a constriction made in the front of the vocal tract?
            'cor',      # [±coronal]. Is the tip or blade of the tongue used to make a constriction?
            'distr',    # [±distributed]. Is a coronal constriction distributed laterally
            'lab',      # [±labial]. Does the segment involve constrictions with or of the lips?
            'hi',       # [±high]. Is the segment produced with the tongue body raised?
            'lo',       # [±low]. Is the segment produced with the tongue body lowered?
            'back',     # [±back]. Is the segment produced with the tongue body in a posterior position?
            'round',    # [±round]. Is the segment produced with the lips rounded?
            'tense'     # [±tense]. Is the segment produced with an advanced tongue root.
        ]

        self.special_tokens = {'<blank>', '<pad>', '<unk>', '<bos>', '<eos>', '|'}

    def get_articulatory_features(self, phonemes: str) -> Dict[str, List[int]]:
        # 1) Remove special tokens
        phonemes_list = phonemes.split()
        phonemes_list = [p for p in phonemes_list if p not in self.special_tokens]
        cleaned_phonemes = ' '.join(phonemes_list)

        # 2) Convert to feature vectors
        feature_matrix = self.ft.word_array(self.feature_names, cleaned_phonemes)

        # 3) Check length is right
        expected_length = len(phonemes_list)
        actual_length = feature_matrix.shape[0]

        if actual_length != expected_length:
            raise ValueError(
                f"Feature matrix length mismatch. "
                f"Expected {expected_length} phonemes, got {actual_length} articulatory features. "
                f"Filtered phonemes: {cleaned_phonemes}"
            )

        # Transpose to dict format
        feature_sequences = {
            name: [int(feature_matrix[i][j]) for i in range(len(feature_matrix))]
            for j, name in enumerate(self.feature_names)
        }
        return feature_sequences