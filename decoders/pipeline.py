from .decoders import CTCGreedyDecoder, CTCBeamSearchDecoder
from utils.logger import init_logger

class DecodingPipeline:
    def __init__(self, tokenizer, decoder_type="greedy",
                 language_model_path=None,
                 beam_size=5,
                 lm_weight=1,
                 word_score=0):
        """
        Args:
            tokenizer: Model's tokenizer
            decoder_type: 'greedy' or 'beam_search'
            **decoder_kwargs: Arguments for specific decoders (beam_size, lm_path, etc.)
        """
        self.tokenizer = tokenizer
        self.decoder_type = decoder_type
        self.logger = init_logger('DecodingPipeline', 'INFO')
        # Initialize the appropriate decoder
        if decoder_type == "greedy":
            self.decoder = CTCGreedyDecoder(tokenizer)
            self.logger.info("Initialized greedy decoder")
        elif decoder_type == "beam_search":
            self.decoder = CTCBeamSearchDecoder(
                tokenizer=tokenizer,
                beam_size=beam_size,
                language_model_path=language_model_path,
                lm_weight=lm_weight,
                word_score=word_score,
                blank_token=tokenizer.word_delimiter_token
            )
            self.logger.info(f"Initialized beam search decoder (beam_size={beam_size})")
            if language_model_path:
                self.logger.info(f"Using language model: {language_model_path}")

        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def decode(self, logits):
        """
        Decode logits to text
        """
        return self.decoder.decode(logits)

    def decode_with_candidates(self, logits):
        """
        Decode logits and return both best results and all candidates (for beam search)
        """
        if self.decoder_type == "beam_search":
            # Beam search returns (best, candidates)
            return self.decode(logits)
        else:
            # Greedy only has one result per sequence
            best_results = self.decode(logits)
            all_candidates = [[result] for result in best_results]
            return best_results, all_candidates

    def __str__(self):
        return f"DecodingPipeline(type={self.decoder_type})"