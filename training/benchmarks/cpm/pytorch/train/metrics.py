import numpy as np
import collections


__all__ = [
    "CorpusLevelScore",
    "average_sentence_level",
    "average_corpus_level",
    "extrema_sentence_level",
    "extrema_corpus_level",
    "greedy_match_corpus_level",
    "greedy_match_sentence_level",
]

_EPSILON = 0.00000000001

# See https://en.wikipedia.org/wiki/1.96 for details of this magic number.
_95_CI_DEVIATE = 1.96

CorpusLevelScore = collections.namedtuple(
    'CorpusLevelScore', ['mean', 'confidence_interval', 'standard_deviation'])


def _compute_corpus_score(scores):
    """
    Compute various statistics from a list of scores.
    The scores come from evaluating a list of sentence pairs.
    The function combines them by mean and standard derivation.

    :param scores: a list of float.
    :return: a CorpusLevelScore.
    """
    return CorpusLevelScore(
        mean=np.mean(scores),
        confidence_interval=_95_CI_DEVIATE * np.std(scores) / len(scores),
        standard_deviation=np.std(scores),
    )


def _cos_sim(a, b):
    """
    Return the cosine similarity of two vector a and b.

    :param a: ndarray of 1D.
    :param b: ndarray of 1D.
    :return: float.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < _EPSILON or b_norm < _EPSILON:
        # zero in, zero out.
        return 0
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


def _embedding_sum(sentence, embeddings):
    """
    Return the sum of embeddings of words in sentence.

    :param sentence: a list of tokens.
    :param embeddings: a KeyedVectors.
    :return: a 1D ndarray of len `embeddings.vector_size`.
    """
    total = sum(_map_to_embeddings(sentence, embeddings))
    return total


def _get_average(sentence, embeddings):
    total = _embedding_sum(sentence, embeddings)
    total_norm = np.linalg.norm(total)
    if total_norm < _EPSILON:
        return np.zeros(embeddings.vector_size)
    return total / total_norm


def average_sentence_level(hypothesis_sentence, reference_sentence,
                           embeddings):
    """
    Compute Average on sentence level.

    :param hypothesis_sentence:
    :param reference_sentence:
    :param embeddings:
    :return:
    """
    return _cos_sim(
        a=_get_average(hypothesis_sentence, embeddings),
        b=_get_average(reference_sentence, embeddings),
    )


def average_corpus_level(hypothesis_corpus, reference_corpus, embeddings,
                         loss_mask):
    """
    Compute Average on corpus level.

    :param hypothesis_corpus: 形状[b,s]
    :param reference_corpus:  形状[b,s]
    :param embeddings:        形状[vocab_size, hidden_size],
    :param loss_mask:         形状[b,s]
    :return:                  形状: 一标量
    """
    assert len(hypothesis_corpus) == len(reference_corpus)
    scores = []

    index = -1
    for hypothesis, reference in zip(hypothesis_corpus, reference_corpus):

        index += 1

        #hypothesis、reference 的形状 [seq_len]
        hypothesis = hypothesis[loss_mask[index] != 0]
        reference = reference[loss_mask[index] != 0]

        #X,Y 的形状 [hidden_size]
        X = _embedding_sum(hypothesis, embeddings)
        Y = _embedding_sum(reference, embeddings)

        # if none of the words in ground truth have embeddings, skip
        if np.linalg.norm(Y) < _EPSILON:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(X) < _EPSILON:
            scores.append(0)
            continue

        # Normalize to unit vectors.
        X /= np.linalg.norm(X)
        Y /= np.linalg.norm(Y)
        scores.append(_cos_sim(X, Y))

    return _compute_corpus_score(scores)


def _get_extrema(vectors):
    """
    Compute the Extrema vector from a list of vectors.

    :param vectors: a list of 1D vectors all having the same shape.
    :return: the Extrema vector.
    """
    max_values = np.max(vectors, axis=0)
    min_values = np.min(vectors, axis=0)
    return np.array([
        min_v if np.abs(min_v) > max_v else max_v
        for min_v, max_v in zip(min_values, max_values)
    ])


def _map_to_embeddings(words, embeddings):
    """
    Map each word in words to its embedding. OOV word maps to zeros.
    Thus the dimension of words may not match that of the returned list.

    :param words: a list of strings.
    :param embeddings: a gensim KeyedVectors.
    :return:  a list of ndarrays.
    """

    def get(word):
        try:
            return embeddings[word]
        except KeyError:
            return np.zeros(embeddings.vector_size)

    return list(map(get, words))


def extrema_sentence_level(hypothesis_sentence, reference_sentence,
                           embeddings):
    """
    Compute Extrema on sentence level.

    :param hypothesis_sentence:
    :param reference_sentence:
    :param embeddings:
    :return:
    """
    hypothesis = _map_to_embeddings(hypothesis_sentence, embeddings)
    reference = _map_to_embeddings(reference_sentence, embeddings)
    return _cos_sim(
        a=_get_extrema(hypothesis),
        b=_get_extrema(reference),
    )


def extrema_corpus_level(hypothesis_corpus, reference_corpus, embeddings):
    """
    Compute Extrema on corpus level.

    :param hypothesis_corpus:
    :param reference_corpus:
    :param embeddings:
    :return:
    """
    scores = []
    for hypothesis, reference in zip(hypothesis_corpus, reference_corpus):
        X = _map_to_embeddings(hypothesis, embeddings)
        Y = _map_to_embeddings(reference, embeddings)

        if np.linalg.norm(X) < _EPSILON:
            continue
        if np.linalg.norm(Y) < _EPSILON:
            scores.append(0)
            continue

        value = _cos_sim(_get_extrema(X), _get_extrema(Y))
        scores.append(value)

    return _compute_corpus_score(scores)


def _greedy_match(a, b):
    """
    Perform the greedy match on two list of word vectors.
    See photos/greedy matching.png.

    :param a: a list of word vectors.
    :param b: a list of word vectors.
    :return: The greedy-matched value.
    """
    sum_max_cosine = sum(max(_cos_sim(a_i, b_i) for b_i in b) for a_i in a)
    if not len(a):
        raise ValueError('empty vector')
    return sum_max_cosine / len(a)


def _greedy_average(a, b):
    """
    Compute the average of greedy matching a on b and b on a.

    :param a: a list of word vectors.
    :param b: a list of word vectors.
    :return: The averaged greedy-matched value.
    """
    # return np.mean([_greedy_match(*args) for args in ((a, b), (b, a))])
    return (_greedy_match(a, b) + _greedy_match(b, a)) / 2


def greedy_match_sentence_level(hypothesis_sentence, reference_sentence,
                                embeddings):
    """
    Compute Greedy Matching on sentence level.

    :param hypothesis_sentence:
    :param reference_sentence:
    :param embeddings:
    :return:
    """
    hyp = _map_to_embeddings(hypothesis_sentence, embeddings)
    ref = _map_to_embeddings(reference_sentence, embeddings)
    return _greedy_average(hyp, ref)


def greedy_match_corpus_level(hypothesis_corpus, reference_corpus, embeddings):
    """
    Compute Greedy Matching on corpus level.

    :param hypothesis_corpus:
    :param reference_corpus:
    :param embeddings:
    :return:
    """
    scores = []
    for hypothesis, reference in zip(hypothesis_corpus, reference_corpus):
        X = _map_to_embeddings(hypothesis, embeddings)
        Y = _map_to_embeddings(reference, embeddings)
        if len(X) == 0 or len(Y) == 0:
            scores.append(0)
            continue
        scores.append(_greedy_average(X, Y))
    return _compute_corpus_score(scores)
