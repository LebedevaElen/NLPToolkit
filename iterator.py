from collections import defaultdict
from typing import List

import numpy as np

class BatchIterator():
    def __init__(self,
                 text_data: List[np.array],
                 target: np.array,
                 max_batch_size: int,
                 extra_features: np.array = None):
        """
        :param text_data: list of tokenized texts, len(text_data) = n_samples
        :param target: np.array of shape (n_samples, target_dim)
        :param max_batch_size: maximum batch size
        :param extra_features: other features
        """

        return

    @property
    def n_batches(self):
        """
        :return: number of batches
        """
        return

    def __iter__(self):
        return


class SameLenBatchIterator():
    """
    Creates batches based on text data length (hence no padding).
    """
    def __init__(self,
                 text_data: List[np.array],
                 target: np.array,
                 max_batch_size: int,
                 extra_features: np.array = None):
        """
        :param text_data: list of tokenized texts, len(text_data) = n_samples
        :param target: np.array of shape (n_samples, target_dim)
        :param max_batch_size: maximum batch size
        :param extra_features: other features
        """
        super().__init__()
        self.text_data = text_data
        self.n_samples = len(self.text_data)
        self.target = target
        if extra_features is None:
            self.n_features = 0
            assert self.n_samples == len(self.target), \
                f'Number of samples in text_data and target ' \
                f'is not the same: ({self.n_samples}, {len(target)})'
        else:
            n_rows, self.n_features = extra_features.shape
            assert n_rows == self.n_samples == len(self.target), \
                f'Number of samples in text_data, target and extra features ' \
                f'is not the same: ' \
                f'({self.n_samples}, {len(self.target)}, {n_rows})'

        self.extra_features = extra_features

        self.max_batch_size = max_batch_size

        self.same_len_bins = defaultdict(list)

        # partition text into same length bins
        for i in range(self.n_samples):
            sent_len = len(self.text_data[i])
            self.same_len_bins[sent_len].append(i)

        # get all text lengths in the data
        self.sent_lens = list(self.same_len_bins.keys())

        # number of batches
        self._n_batches = sum(
            [len(self.same_len_bins[sent_len]) // max_batch_size
             + int(len(self.same_len_bins[sent_len]) % max_batch_size > 0)
             for sent_len in self.same_len_bins])

    @property
    def n_batches(self):
        """
        :return: number of batches
        """
        return self._n_batches

    def __iter__(self):
        """
        Iterates over same length batches
        """
        np.random.shuffle(self.sent_lens)
        # iterating over same length texts
        for sent_len in self.sent_lens:
            n_len_samples = len(self.same_len_bins[sent_len])
            indices = np.arange(n_len_samples)
            np.random.shuffle(indices)
            # iterating over batches with sent_len length
            for start in range(0, n_len_samples, self.max_batch_size):
                end = min(start + self.max_batch_size, n_len_samples)
                batch_indices = indices[start:end]
                if len(batch_indices) == 1:
                    continue
                text_batch = np.zeros((len(batch_indices), sent_len))
                features_batch = None
                if self.n_features > 0:
                    features_batch = np.zeros((len(batch_indices),
                                               self.n_features))
                target_batch = np.zeros((len(batch_indices), 1))

                for batch_ind, sample_ind in enumerate(batch_indices):
                    text_batch[batch_ind, :] = self.text_data[
                        self.same_len_bins[sent_len][sample_ind]]
                    if self.n_features > 0:
                        features_batch[batch_ind, :] = self.extra_features[
                            self.same_len_bins[sent_len][sample_ind]]
                    target_batch[batch_ind, 0] = self.target[
                        self.same_len_bins[sent_len][sample_ind]]

                yield text_batch, target_batch, features_batch
