"""
Phoneme n-gram language model for beam search rescoring.

The fly swarm proposes top-k candidates per slot; this LM provides
sequence-level rescoring to pick coherent phoneme sequences.

Biology analogy: this is the "arbitrator" layer — like downstream
neurons that integrate multiple MBON outputs to make a final decision.
Not replacing the MB specialists, but coordinating their proposals.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class BeamState:
    """State in beam search."""
    phonemes: List[str]  # Phoneme sequence so far
    score: float         # Combined score (fly + LM)
    lm_score: float      # LM score component
    fly_score: float     # Fly score component
    
    def __lt__(self, other):
        return self.score > other.score  # Higher score = better


class PhonemeNGramLM:
    """
    Simple n-gram language model over phoneme sequences.
    
    Trained on CMUdict pronunciations to capture common phoneme patterns.
    Used to rescore top-k fly proposals via beam search.
    """
    
    def __init__(self, n: int = 3):
        """
        Args:
            n: N-gram order (2=bigram, 3=trigram)
        """
        self.n = n
        self.START = '<s>'
        self.END = '</s>'
        
        # Counts: context -> next_phoneme -> count
        self.counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.context_totals: Dict[Tuple[str, ...], int] = defaultdict(int)
        
        # Vocabulary
        self.vocab: set = set()
        
        # Smoothing
        self.smoothing = 0.1  # Add-k smoothing
        
        self.is_fitted = False
    
    def fit(self, pronunciations: List[List[str]], verbose: bool = True):
        """
        Fit n-gram model on phoneme sequences.
        
        Args:
            pronunciations: List of phoneme sequences (e.g., [['K', 'AE', 'T'], ...])
            verbose: Print progress
        """
        if verbose:
            print(f"Fitting {self.n}-gram phoneme LM on {len(pronunciations)} sequences...")
        
        for phones in pronunciations:
            # Strip stress markers
            phones = [self._strip_stress(p) for p in phones]
            self.vocab.update(phones)
            
            # Add start/end tokens
            padded = [self.START] * (self.n - 1) + phones + [self.END]
            
            # Count n-grams
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                next_phone = padded[i + self.n - 1]
                
                self.counts[context][next_phone] += 1
                self.context_totals[context] += 1
        
        self.is_fitted = True
        
        if verbose:
            print(f"  Vocabulary: {len(self.vocab)} phonemes")
            print(f"  Contexts: {len(self.counts)}")
            total_ngrams = sum(self.context_totals.values())
            print(f"  Total {self.n}-grams: {total_ngrams}")
    
    def _strip_stress(self, phoneme: str) -> str:
        """Strip stress markers from phoneme."""
        return ''.join(c for c in phoneme if not c.isdigit())
    
    def log_prob(self, phoneme: str, context: List[str]) -> float:
        """
        Get log probability of phoneme given context.
        
        Args:
            phoneme: Next phoneme
            context: Previous (n-1) phonemes
        
        Returns:
            Log probability (natural log)
        """
        phoneme = self._strip_stress(phoneme)
        context = [self._strip_stress(p) for p in context]
        
        # Pad context if needed
        while len(context) < self.n - 1:
            context = [self.START] + context
        
        # Take last (n-1) tokens as context
        context_tuple = tuple(context[-(self.n - 1):])
        
        # Add-k smoothing
        count = self.counts[context_tuple].get(phoneme, 0) + self.smoothing
        total = self.context_totals[context_tuple] + self.smoothing * (len(self.vocab) + 1)
        
        if total <= 0:
            # Fallback: uniform distribution
            return np.log(1.0 / max(len(self.vocab), 1))
        
        return np.log(count / total)
    
    def score_sequence(self, phonemes: List[str]) -> float:
        """Score a complete phoneme sequence."""
        phonemes = [self._strip_stress(p) for p in phonemes]
        context = [self.START] * (self.n - 1)
        
        total_log_prob = 0.0
        for phone in phonemes + [self.END]:
            total_log_prob += self.log_prob(phone, context)
            context = context[1:] + [phone]
        
        return total_log_prob
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            'n': self.n,
            'counts': {str(k): dict(v) for k, v in self.counts.items()},
            'context_totals': {str(k): v for k, v in self.context_totals.items()},
            'vocab': list(self.vocab),
            'smoothing': self.smoothing,
            'is_fitted': self.is_fitted,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PhonemeNGramLM':
        """Deserialize from dict."""
        lm = cls(n=d['n'])
        lm.counts = defaultdict(lambda: defaultdict(int))
        for k, v in d.get('counts', {}).items():
            # Convert string key back to tuple
            key = eval(k) if k.startswith('(') else (k,)
            lm.counts[key] = defaultdict(int, v)
        lm.context_totals = defaultdict(int)
        for k, v in d.get('context_totals', {}).items():
            key = eval(k) if k.startswith('(') else (k,)
            lm.context_totals[key] = v
        lm.vocab = set(d.get('vocab', []))
        lm.smoothing = d.get('smoothing', 0.1)
        lm.is_fitted = d.get('is_fitted', False)
        return lm
    
    def save(self, path: str):
        """Save LM to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)
        print(f"Phoneme LM saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PhonemeNGramLM':
        """Load LM from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        lm = cls.from_dict(d)
        print(f"Phoneme LM loaded from {path} ({lm.n}-gram, {len(lm.vocab)} phonemes)")
        return lm


def build_phoneme_lm(
    n: int = 3,
    max_words: int = None,
    seed: int = 42,
    verbose: bool = True,
) -> PhonemeNGramLM:
    """
    Build phoneme n-gram LM from CMUdict.
    
    Args:
        n: N-gram order
        max_words: Limit training words (None = all)
        seed: Random seed for sampling
        verbose: Print progress
    """
    from .lexicon import get_lexicon
    
    lexicon = get_lexicon()
    pronunciations = list(lexicon.values())
    
    if max_words and len(pronunciations) > max_words:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(pronunciations), size=max_words, replace=False)
        pronunciations = [pronunciations[i] for i in indices]
    
    lm = PhonemeNGramLM(n=n)
    lm.fit(pronunciations, verbose=verbose)
    
    return lm


class BeamSearchDecoder:
    """
    Beam search decoder that combines fly swarm scores with LM rescoring.
    
    The swarm proposes top-k candidates per slot; beam search finds the
    best sequence considering both fly confidence and phoneme LM probability.
    """
    
    def __init__(
        self,
        swarm,  # FlySwarm instance
        lm: PhonemeNGramLM,
        beam_width: int = 5,
        lm_weight: float = 0.3,
        top_k: int = 5,
    ):
        """
        Args:
            swarm: FlySwarm for getting candidate scores
            lm: Phoneme n-gram LM
            beam_width: Number of hypotheses to keep at each step
            lm_weight: Weight for LM score (vs fly score)
            top_k: Number of candidates to consider per slot
        """
        self.swarm = swarm
        self.lm = lm
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.top_k = top_k
    
    def decode(
        self,
        word: str,
        n_phonemes: int,
        context_size: int = None,
        vote_strategy: str = None,
    ) -> Tuple[List[str], float]:
        """
        Decode word to phoneme sequence using beam search.
        
        Args:
            word: Input word
            n_phonemes: Number of phonemes to predict
            context_size: Letter context size
            vote_strategy: Voting strategy for getting scores
        
        Returns:
            best_phonemes: Best phoneme sequence
            best_score: Combined score
        """
        if context_size is None:
            context_size = self.swarm.config.specialist_config.context_size
        
        word = word.lower()
        n_letters = len(word)
        
        # Initialize beam with empty sequence
        beam = [BeamState(phonemes=[], score=0.0, lm_score=0.0, fly_score=0.0)]
        
        for p_idx in range(n_phonemes):
            # Get letter context for this position
            if n_phonemes == 1:
                letter_pos = n_letters // 2
            else:
                letter_pos = int(round(p_idx * (n_letters - 1) / (n_phonemes - 1)))
            letter_pos = max(0, min(letter_pos, n_letters - 1))
            
            context_chars = []
            for offset in range(-context_size, context_size + 1):
                idx = letter_pos + offset
                if 0 <= idx < n_letters:
                    context_chars.append(word[idx])
                else:
                    context_chars.append('_')
            letter_context = ''.join(context_chars)
            
            # Get top-k candidates from swarm
            candidates = self._get_top_k_candidates(letter_context, vote_strategy)
            
            # Expand beam
            new_beam = []
            for state in beam:
                for phoneme, fly_score in candidates:
                    # Compute LM score increment
                    lm_score_inc = self.lm.log_prob(phoneme, state.phonemes)
                    
                    # Combined score: fly + lambda * LM
                    # Normalize fly_score to log-like range
                    fly_log = fly_score * 2.0  # Scale factor
                    combined_inc = fly_log + self.lm_weight * lm_score_inc
                    
                    new_state = BeamState(
                        phonemes=state.phonemes + [phoneme],
                        score=state.score + combined_inc,
                        lm_score=state.lm_score + lm_score_inc,
                        fly_score=state.fly_score + fly_log,
                    )
                    new_beam.append(new_state)
            
            # Prune to beam width
            new_beam.sort(key=lambda s: -s.score)
            beam = new_beam[:self.beam_width]
        
        # Add end-of-sequence LM score
        for state in beam:
            end_score = self.lm.log_prob(self.lm.END, state.phonemes)
            state.lm_score += end_score
            state.score += self.lm_weight * end_score
        
        # Return best hypothesis
        beam.sort(key=lambda s: -s.score)
        best = beam[0]
        
        return best.phonemes, best.score
    
    def _get_top_k_candidates(
        self,
        letter_context: str,
        vote_strategy: str = None,
    ) -> List[Tuple[str, float]]:
        """Get top-k phoneme candidates with scores."""
        # Get raw scores from swarm
        scores = self.swarm._get_raw_scores(letter_context)
        
        # Apply calibration if available and requested
        if vote_strategy == 'calibrated' and self.swarm.calibrator.is_fitted:
            scores = self.swarm.calibrator.calibrate(scores)
        
        # Sort by score and take top-k
        sorted_candidates = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_candidates[:self.top_k]


def predict_word_with_beam(
    swarm,
    word: str,
    n_phonemes: int,
    lm: PhonemeNGramLM,
    beam_width: int = 5,
    lm_weight: float = 0.3,
    top_k: int = 5,
    vote_strategy: str = None,
) -> List[str]:
    """
    Convenience function to predict word with beam search.
    
    Args:
        swarm: FlySwarm instance
        word: Input word
        n_phonemes: Number of phonemes
        lm: Phoneme LM
        beam_width: Beam width
        lm_weight: LM weight
        top_k: Candidates per slot
        vote_strategy: Base voting strategy for scores
    
    Returns:
        Predicted phoneme sequence
    """
    decoder = BeamSearchDecoder(
        swarm=swarm,
        lm=lm,
        beam_width=beam_width,
        lm_weight=lm_weight,
        top_k=top_k,
    )
    phonemes, _ = decoder.decode(word, n_phonemes, vote_strategy=vote_strategy)
    return phonemes


if __name__ == "__main__":
    # Test LM building
    print("Testing phoneme n-gram LM...")
    
    lm = build_phoneme_lm(n=3, max_words=5000)
    
    # Test scoring
    test_seqs = [
        ['K', 'AE', 'T'],       # cat - common
        ['TH', 'AO', 'T'],      # thought - common
        ['ZH', 'ZH', 'ZH'],     # unlikely
    ]
    
    print("\nSequence scores:")
    for seq in test_seqs:
        score = lm.score_sequence(seq)
        print(f"  {' '.join(seq):20} log_prob={score:.3f}")
    
    # Test conditional probabilities
    print("\nConditional probs after 'K':")
    context = ['K']
    for phone in ['AE', 'IY', 'AA', 'ZH']:
        log_p = lm.log_prob(phone, context)
        print(f"  P({phone}|K) = {np.exp(log_p):.4f}")
