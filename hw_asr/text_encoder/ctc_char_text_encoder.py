from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"
    EMPTY_IND = 0

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = self.EMPTY_IND
        decoded_input = []
        for ind in inds:
            if ind == last_char:
                continue
            if ind != self.EMPTY_IND:
                decoded_input += [self.ind2char[ind]]
            last_char = ind
        return ''.join(decoded_input)

    # def ctc_beam_search(self, probs, probs_length,
    #                     beam_size: int = 100) -> List[Hypothesis]:
    #     """
    #     Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
    #     """
    #     def extend_and_merge(dp, proba):
    #         new_dp = defaultdict(float)
    #         for (res, last_char), v in dp.items():
    #             for i in range(len(proba)):
    #                 if self.ind2char[i] == last_char:
    #                     new_dp[(res, last_char)] += v * proba[i]
    #                 else:
    #                     new_dp[(res + last_char.replace(self.EMPTY_TOK, ''), self.ind2char[i])] += v * proba[i]
    #
    #         return new_dp
    #
    #     def cut_beams(dp: dict, beam_size: int) -> dict:
    #         return dict(list(sorted(dp.items(), key=lambda x: x[1]))[-beam_size:])
    #
    #     print(probs.shape)
    #     # print(probs)
    #     # print(self.ind2char[0])
    #     assert len(probs.shape) == 2
    #     char_length, voc_size = probs.shape
    #     assert voc_size == len(self.ind2char)
    #     hypos: List[Hypothesis] = []
    #
    #     dp = {
    #         ('', self.EMPTY_TOK): 1.0
    #     }
    #     for prob in probs:
    #         dp = extend_and_merge(dp, prob)
    #         dp = cut_beams(dp, beam_size)
    #     hypos = [Hypothesis((res+last_char).strip().replace(self.EMPTY_TOK, ''), proba) for (res, last_char), proba in dp.items()]
    #
    #     return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def ctc_beam_search(self, probs, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        def extend_and_merge(next_char_probs, src_paths):
            new_paths = defaultdict(float)
            for next_char_ind, next_char_prob in enumerate(next_char_probs):
                next_char = self.ind2char[next_char_ind]
                for (text, last_char), path_prob in src_paths.items():
                    new_prefix = text if next_char == last_char else (text + next_char)
                    new_prefix = new_prefix.replace(self.EMPTY_TOK, '')
                    new_paths[(new_prefix, next_char)] += path_prob * next_char_prob
            return new_paths

        def cut_beams(paths, beam_size):
            return dict(sorted(paths.items(), key=lambda x: x[1])[-beam_size:])

        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        paths = {('', self.EMPTY_TOK): 1.0}
        for prob in probs:
            paths = extend_and_merge(prob, paths)
            paths = cut_beams(paths, beam_size)

        return [Hypothesis(prefix, score) for (prefix, _), score in sorted(paths.items(), key=lambda x: -x[1])]
