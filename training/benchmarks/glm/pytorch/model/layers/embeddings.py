import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class VocabEmbedding(torch.nn.Module):
    """Embedding in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings,
                                             self.embedding_dim))

        # initialize.
        init_method(self.weight)

    def forward(self, input_):
        # Get the embeddings.
        output = F.embedding(input_, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        return output

if __name__=="__main__":
    num_embeddings = 10
    embedding_dim = 3
    layer = VocabEmbedding(num_embeddings, embedding_dim)
    input_ = torch.tensor([[1, 2, 3]])
    print(layer(input_))
    print(layer.weight)

