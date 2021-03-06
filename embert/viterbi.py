#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Sequence

import numpy as np


class Viterbi:
    """Implements the Viterbi algorithm."""
    def __init__(self, init: List[float], trans: List[List[float]],
                 emission: List[List[float]] = None, logp: bool = False):
        """
        Initializes the basic probabilities.

        .. note::
        _emission_ can be ``None``, but it should not be for regular Viterbi.

        :param init: Initial state probabilities.
        :param trans: Transition probabilities: ``trans[i, j] = P(s_j|s_i)``.
        :param emission: The emission probabilities:
                         ``emission[i, j] = P(w_j|s_i)``.
        :param logp: if ``True``, the arrays above already contain logarithmic
                     probabilities, so no conversion takes place; otherwise,
                     the arrays are presumed to contain straight-up
                     probabilities, which are then converted to logprobs.
        """
        fn = np.array if logp else np.log
        self.init = fn(init, dtype=float)
        # Trans is transposed, as that's how we use it
        self.trans = fn(trans, dtype=float).T
        self.emission = fn(emission, dtype=float) if emission else None

    def __call__(self, observations: Sequence[int]) -> List[int]:
        """
        Execute the algorithm based on _observations_.

        :param observations: the observations; a list of integers.
        :return: the most probable state (integer) sequence.
        """
        return self.viterbi_inner(self.emission, observations)

    def num_states(self) -> int:
        """Returns the number of hidden states."""
        return len(self.init)

    def viterbi_inner(self, emission, observations) -> List[int]:
        """
        The inner workings of the algorithm. So that it can be used by both
        regular and reversed Viterbis, the emissions are not taken from
        the class but are passed from :meth:`Viterbi.__call__`.

        :param emission: the emission probabilities.
        :param observations: a list of integers that index into the columns
                             of _emission_. For regular Viterbi, these are the
                             of the observed tokens.
        :return: the most probable state (integer) sequence.
        """
        t1 = np.zeros((len(self.init), len(observations)), dtype=float)
        t2 = np.zeros(t1.shape, dtype=np.uint16)
        t1[:, 0] = self.init + emission[:, observations[0]]
        max_idx1 = np.arange(len(self.trans))
        for idx in range(1, len(observations)):
            curr = self.trans + t1[:, idx - 1]
            max_idx2 = np.argmax(curr, axis=1)
            maxs = curr[max_idx1, max_idx2]
            t1[:, idx] = maxs + emission[:, observations[idx]]
            t2[:, idx] = max_idx2

        states = [0] * len(observations)
        states[-1] = np.argmax(t1[:, -1], axis=0)
        for i in range(len(states) - 2, -1, -1):
            states[i] = t2[:, i + 1][states[i + 1]]
        return states


class ReverseViterbi(Viterbi):
    """
    Implements a so-called reverse Viterbi algorithm. In this problem, we
    do not know the emission probabilities. Instead, for each time step, we
    are told the probabilities of each hidden state. The task is still to
    find the most probable hidden state sequence.

    The algorithm below basically replaces the emission probabilities with
    the pre-step hidden state distribution. I believe this is mathematically
    not sound, but should probably work -- if for nothing else, at least to
    rule out impossible tag sequences in token classification tasks.
    """
    def __init__(self, init, trans, logp: bool = False):
        """Same as with the regular :class:`Viterbi`, only with no emissions."""
        super().__init__(init, trans, logp=logp)

    def __call__(
        self, state_probs: List[List[float]], logp: bool = False
    ) -> List[int]:
        """
        Runs the algorithm with the specified per-time-step hidden state
        probabilities.

        :param state_probs: the hidden state probabilities:
                            ``state_probs[i, j] == P_j(s_i).``
        :param logp: if ``True``, _state_probs_ already contains logarithmic
                     probabilities, so no conversion takes place; otherwise,
                     it is presumed to contain straight-up
                     probabilities, which are then converted to logprobs.
        :return: the most probable state (integer) sequence.
        """
        fn = np.array if logp else np.log
        return self.viterbi_inner(fn(state_probs),
                                  list(range(state_probs.shape[1])))


if __name__ == '__main__':
    toy = Viterbi([0.5, 0.5],
                  [[0.5, 0.5], [0.4, 0.6]],
                  [[0.2, 0.3, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]])
    assert toy([2, 2, 1, 0, 1, 3, 2, 0, 0]) == [0, 0, 0, 1, 1, 1, 1, 1, 1]

    stanford = Viterbi([0.6, 0.4],
                       [[0.7, 0.3], [0.4, 0.6]],
                       [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    assert stanford([0, 1, 2]) == [0, 0, 1]

    nap_eso = ReverseViterbi([0.7, 0.3],
                             [[0.8, 0.2], [0.3, 0.7]])
    assert nap_eso(np.array([[0.7, 0.3], [0.1, 0.9], [0.5, 0.5]], dtype=float).T) == [0, 1, 1]
