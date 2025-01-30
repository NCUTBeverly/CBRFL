import numpy as np
import six
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from config.env import DATASET


def get_glove(dim=25):
    return GloVe(name='twitter.27B', dim=dim, cache=f'{DATASET}/glove/')


def pad_sequences(sequences, max_len=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):

    """Pads each sequence to the same length: the length of the longest"""

    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')

    num_samples, lengths, sample_shape, flag = len(sequences), [], (), True

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables.')

    if max_len is None:
        max_len = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, max_len) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        # empty list/array was found
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s'
                             % (trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


class ToFixedSeq:

    def __init__(self, max_len, dim):
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        assert dim in [25, 50, 100, 200], 'dim = (25,50,100,200)'
        self.glove = get_glove(dim)
        self.max_len = max_len

    def __call__(self, sentence):
        tokens = self.tokenizer(sentence)
        vecs = self.glove.get_vecs_by_tokens(tokens)
        seq = pad_sequences([vecs], max_len=self.max_len, dtype='float32', padding='post')
        return seq


