import torch
import numpy as np
from torch.autograd import Variable

from .common_utils import Vocab, GlobalNames

__all__ = [
    'tile_batch',
    'mask_scores',
    'tensor_gather_helper',
    'reranking_beams'
]

_FLOAT32_INF = np.float32(np.finfo('float32').max / 10)

def tile_batch(x, multiplier, batch_dim=0):
    """
    :type x: Variable
    """
    x_size = x.size()
    out_size = x_size[:batch_dim] + (x_size[batch_dim] * multiplier,) + x_size[batch_dim+1:]

    x_tiled = torch.unsqueeze(x, dim=batch_dim + 1)
    x_tiled = x_tiled.repeat(*[1 if d != batch_dim + 1 else multiplier for d in range(len(x_size) + 1)])
    x_tiled = x_tiled.view(*out_size)

    return x_tiled

def mask_scores(scores, beam_mask):

    """
    :type scores: Variable
    :param scores: [B, Bm, N]

    :type beam_mask: Variable, 1 means closed beam
    :param beam_mask: [B, Bm]
    """

    vocab_size = scores.size(-1)

    finished_row = beam_mask.new(vocab_size, ).zero_().float() + float(_FLOAT32_INF) # [N, ]
    finished_row[Vocab.EOS] = 0.0
    scores += torch.matmul(torch.unsqueeze(beam_mask.float(), 2), torch.unsqueeze(finished_row, 0))

    return scores


def tensor_gather_helper(gather_indices,
                         gather_from,
                         batch_size,
                         beam_size,
                         gather_shape):

    range_ = (torch.arange(0, batch_size) * beam_size).long()
    if GlobalNames.USE_GPU:
        range_ = range_.cuda()

    gather_indices_ = (gather_indices + torch.unsqueeze(range_, 1)).view(-1)

    output = torch.index_select(gather_from.view(*gather_shape), 0, gather_indices_)

    out_size = gather_from.size()[:1 + len(gather_shape)]

    return output.view(*out_size)

def reranking_beams(word_ids, scores):

    word_ids = word_ids.cpu().numpy()
    scores = scores.cpu().numpy()

    # Reranking beams
    reranked_beams = np.argsort(scores, axis=1)
    reranked_word_ids = np.ones_like(word_ids) * Vocab.EOS

    for b in range(scores.shape[0]):
        for ii in reranked_beams[b]:
            reranked_word_ids[b, ii] = word_ids[b, ii]

    # Trim BOS
    reranked_word_ids = reranked_word_ids.tolist()

    return reranked_word_ids