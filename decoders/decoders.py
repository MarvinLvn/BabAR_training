import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torchaudio.models.decoder import ctc_decoder
from utils.logger import init_logger
import os
import re

class CTCGreedyDecoder:
    """
    Greedy CTC decoder - takes the most likely token at each timestep
    """
    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: The tokenizer used for encoding/decoding
        """
        self.tokenizer = tokenizer
        self.logger = init_logger('CTCGreedyDecoder', 'INFO')

    def decode(self,
               logits: torch.Tensor) -> List[int]:
        """
        Decode logits using greedy decoding - fully batched

        Args:
            logits: [batch_size, seq_len, vocab_size] raw logits

        Returns:
            List of decoded strings
        """
        # Get most likely tokens for entire batch
        predicted_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        results = self.tokenizer.batch_decode(predicted_ids)
        return results


class CTCBeamSearchDecoder:
    def __init__(self,
                 tokenizer,
                 language_model_path: str,
                 beam_size: int = 5,
                 lm_weight: float = 1,
                 word_score: float = 0.0,
                 blank_token = "<blank>"):
        """
        Args:
            tokenizer: The tokenizer used for encoding/decoding
            beam_size: Number of beams for beam search
            language_model_path: Path to KenLM language model (.arpa or .klm)
            lm_weight: Weight for language model scores
            word_score: Score added for each word
            blank_token: CTC blank token
        """
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.blank_token = blank_token
        self.logger = init_logger('CTCBeamSearchDecoder', 'INFO')

        # Get vocabulary and create token mapping
        self.vocab = self.tokenizer.get_vocab()
        self.tokens = [None] * len(self.vocab)
        for token, idx in self.vocab.items():
            self.tokens[idx] = token

        # Get blank token ID
        self.blank_id = self.vocab.get(blank_token, 0)
        self.logger.info(f"Blank token: '{blank_token}' (ID: {self.blank_id})")

        # Initialize the decoder
        self._setup_decoder(language_model_path)

    def _setup_decoder(self, language_model_path: str):
        self.logger.info(f"Loading language model from: {language_model_path}")
        if not os.path.exists(language_model_path):
            raise ValueError(f"Language model not found: {language_model_path}")

        if self.word_score != 0:
            special_tokens = [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token,
                              self.tokenizer.unk_token, self.tokenizer.cls_token, self.tokenizer.sep_token]
            lexicon = [p for p in self.tokenizer.get_vocab() if p not in special_tokens]
            lexicon_path = os.path.join(os.path.dirname(language_model_path), 'lexicon.txt')
            if not os.path.isfile(lexicon_path):
                with open(lexicon_path, 'w') as f:
                    f.writelines(f"{p} {p}\n" for p in lexicon)
        else:
            lexicon_path = None
        self.decoder = ctc_decoder(
            lexicon=lexicon_path,  # We'll use None for phoneme-level decoding
            tokens=self.tokens,
            lm=language_model_path,
            nbest=self.beam_size,
            beam_size=self.beam_size,
            lm_weight=self.lm_weight,
            word_score=self.word_score,
            unk_score=float('-inf'),  # Penalty for unknown tokens
            sil_score=0.0,  # Score for silence
            log_add=False,
            blank_token=self.blank_token,
            sil_token=self.blank_token,
            unk_word=self.blank_token,
        )

    def decode(self,
               logits: torch.Tensor) -> Tuple[List[str], List[List[str]]]:
        """
        Decode logits using beam search with native batch support

        Args:
            logits: [batch_size, seq_len, vocab_size] raw logits
        Returns:
            Tuple of:
            - List of best decoded strings (one per batch item)
            - List of lists containing all candidates (beam_size candidates per batch item)
        """
        log_probs = F.log_softmax(logits, dim=-1).cpu()  # [batch_size, seq_len, vocab_size]
        beam_results = self.decoder(log_probs)

        # Extract results for each batch item
        best_results = []
        all_candidates = []
        for batch_idx, batch_results in enumerate(beam_results):
            batch_candidates = []
            # Process all candidates for this batch item
            for candidate_idx, candidate in enumerate(batch_results):
                # Extract tokens and convert to string
                tokens = [self.tokens[token_idx] for token_idx in candidate.tokens]
                decoded_text = ' '.join(tokens)
                batch_candidates.append((decoded_text, candidate.score))

            # Best result is the first candidate (highest score)
            best_results.append(batch_candidates[0][0] if batch_candidates else "")
            all_candidates.append(batch_candidates)
        return best_results, all_candidates

