import torch
import math

from .transformer_block import GLMTransformerLayer
from .layernorm import LayerNorm
from model.models.checkpoint import checkpoint


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


class GLMTransformer(torch.nn.Module):
    """GPT-2 transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """

    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        max_sequence_length,
        max_memory_length,
        embedding_dropout_prob,
        attention_dropout_prob,
        output_dropout_prob,
        checkpoint_activations,
        checkpoint_num_layers=1,
        layernorm_epsilon=1.0e-5,
        init_method_std=0.02,
        use_scaled_init_for_output_weights=True,
        block_position_encoding=True,
        attention_scale=1.0,
    ):
        super(GLMTransformer, self).__init__()
        self.hidden_size = hidden_size
        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(
                init_method_std, num_layers)
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.block_position_encoding = block_position_encoding

        # Position embedding (serial).
        if block_position_encoding:
            self.position_embeddings = torch.nn.Embedding(
                max_sequence_length + 1, hidden_size)
            self.block_position_embeddings = torch.nn.Embedding(
                max_sequence_length + 1, hidden_size)
            torch.nn.init.normal_(self.block_position_embeddings.weight,
                                  mean=0.0,
                                  std=init_method_std)
        else:
            self.position_embeddings = torch.nn.Embedding(
                max_sequence_length, hidden_size)
        # Initialize the position embeddings.
        torch.nn.init.normal_(self.position_embeddings.weight,
                              mean=0.0,
                              std=init_method_std)

        def get_layer():

            return GLMTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                output_layer_init_method=output_layer_init_method,
                relative_encoding=False,
                performer=False,
                attention_scale=attention_scale)

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer() for _ in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # if deepspeed.checkpointing.is_configured():
        #     global get_cuda_rng_tracker, checkpoint
        #     get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
        #     checkpoint = deepspeed.checkpointing.checkpoint

    def forward(self,
                hidden_states,
                position_ids,
                attention_mask,
                memory_states=None,
                encoder_states=None,
                return_memory=False,
                detach_memory=True):
        batch_size, query_length = hidden_states.size()[:2]
        memory_length = memory_states[0].size(1) if memory_states else 0
        key_length = query_length + memory_length
        # attention mask is the beginning postion of B region, \in [0, query_len)
        is_scalar = torch.numel(attention_mask) == 1
        is_sep = is_scalar or torch.numel(attention_mask) == batch_size

        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            # conventional transformer
            def build_mask_matrix(seq_length, sep, memory_length=0):
                m = hidden_states.new_ones((1, seq_length, seq_length))
                m = torch.tril(m)
                if is_scalar:
                    m[0, :, :sep] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = torch.arange(seq_length,
                                       device=sep.device,
                                       dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                if memory_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = torch.cat((hidden_states.new_ones(
                        (batch_size, seq_length, memory_length)), m),
                                  dim=2)
                m = m.unsqueeze(1)
                return m

            attention_mask = build_mask_matrix(query_length,
                                               sep,
                                               memory_length=memory_length)
        else:
            attention_mask = attention_mask[:, :, :,
                                            -query_length - memory_length:]

        if self.block_position_encoding:
            position_ids, block_position_ids = position_ids[:,
                                                            0], position_ids[:,
                                                                             1]
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        if self.block_position_encoding:
            block_position_embeddings = self.block_position_embeddings(
                block_position_ids)
            hidden_states = hidden_states + block_position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        def check_detach(_hidden_states):
            if detach_memory:
                return _hidden_states.detach()
            return _hidden_states

        if self.max_memory_length > 0 or return_memory:
            mem_layers = [check_detach(hidden_states)]
        else:
            mem_layers = []

        def custom(start, end):

            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_, inputs = inputs[0], inputs[1:]

                inputs, mems_ = inputs[:1], inputs[1:]
                for i, layer in enumerate(layers_):
                    mem_i_ = mems_[i] if mems_ else None
                    x_ = layer(x_, *inputs, mem=mem_i_)
                    if self.max_memory_length > 0 or return_memory:
                        mem_layers.append(check_detach(x_))
                return x_

            return custom_forward

        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                args = [hidden_states, attention_mask]
                if memory_states:
                    args += memory_states[l:l + chunk_length]
                hidden_states = checkpoint(custom(l, l + chunk_length), *args)
                l += chunk_length
        else:
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask]
                mem_i = memory_states[i] if memory_states else None
                hidden_states = layer(*args, mem=mem_i)
                if self.max_memory_length > 0 or return_memory:
                    mem_layers.append(check_detach(hidden_states))

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if self.max_memory_length > 0 or return_memory:
            mem_layers = self.update_mems(mem_layers,
                                          memory_states,
                                          return_memory=return_memory)

        return (output, mem_layers)

    def update_mems(self, hiddens, mems, return_memory=False):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length
        if not return_memory:
            new_memory_length = min(self.max_memory_length, new_memory_length)
        new_mems = []
        # with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(
                    torch.cat((mems[i][:, -new_memory_length + query_length:],
                               hiddens[i]),
                              dim=1))
        return new_mems


if __name__ == "__main__":

    batch_size = 2
    seq_len = 512
    hidden_size = 1024
    hidden_states = torch.rand([batch_size, seq_len, hidden_size],
                               dtype=torch.float32).to("cuda")
    position_ids = torch.ones([batch_size, 2, seq_len],
                              dtype=torch.int64).to('cuda')
    attention_mask = torch.tensor([5, 10]).to('cuda')

    model = GLMTransformer(num_layers=24,
                           hidden_size=1024,
                           num_attention_heads=16,
                           max_sequence_length=512,
                           max_memory_length=0,
                           embedding_dropout_prob=0.1,
                           attention_dropout_prob=0.1,
                           output_dropout_prob=0.1,
                           checkpoint_activations=True,
                           checkpoint_num_layers=1,
                           layernorm_epsilon=1.0e-5,
                           init_method_std=0.02,
                           use_scaled_init_for_output_weights=True,
                           block_position_encoding=True,
                           attention_scale=1.0).to('cuda')

    outputs = model(hidden_states, position_ids, attention_mask)
    print(outputs[0].shape)
    print(outputs[1])
