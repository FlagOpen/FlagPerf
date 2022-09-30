import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from model.models.transformer import GLMTransformer
from model.layers.embeddings import VocabEmbedding

FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)

def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GLMModel(torch.nn.Module):
    """GLM Language model.

    The output of the forward method are the logits (parallel or
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 block_position_encoding=True,
                 output_predict=True,
                 attention_scale=1.0,
                 ):

        super(GLMModel, self).__init__()

        self.output_predict = output_predict
        self.hidden_size = hidden_size

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = VocabEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Transformer
        self.transformer = GLMTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       max_sequence_length,
                                                       max_memory_length,
                                                       embedding_dropout_prob,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers,
                                                       attention_scale=attention_scale,
                                                       block_position_encoding=block_position_encoding)
 

    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = "Freeze transformer"
        self.word_embeddings.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if tune_prefix_layers is not None:
            log_str += f" tune {tune_prefix_layers} prefix layers"
            for i in range(tune_prefix_layers):
                self.transformer.layers[i].requires_grad_(True)
        print(log_str)

    def forward(self, input_ids, position_ids, attention_mask, *mems, return_memory=False, detach_memory=True,
                prompt_pos=None):
        # Embeddings.
        batch_size = input_ids.size(0)
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = torch.arange(
                batch_size, device=input_ids.device).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds
        # Transformer.
        transformer_output = self.transformer(embeddings, position_ids, attention_mask, mems,
                                              return_memory=return_memory, detach_memory=detach_memory)
        logits, hidden_layers = transformer_output
        outputs = hidden_layers

        if self.output_predict:
            # Parallel logits.
            logits = F.linear(logits, self.word_embeddings.weight)
   
        return (logits, *outputs)


class GLMForMultiTokenCloze(torch.nn.Module):
    def __init__(self, language_model: GLMModel, take_softmax=True, length_penalty=0.0):
        super(GLMForMultiTokenCloze, self).__init__()
        self.model = language_model
        self.take_softmax = take_softmax
        self.length_penalty = length_penalty

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # [h.remove() for h in self.hook_handles]
        sd = self.model.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def forward(self, input_ids, position_ids, attention_mask, target_ids=None, logit_mask=None, prompt_pos=None):
        if target_ids == None:
            return self.model(input_ids, position_ids, attention_mask)
        num_choices = None
        if len(input_ids.shape) == 3:
            batch_size, num_choices = input_ids.shape[:2]
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, *attention_mask.size()[2:])
            position_ids = position_ids.reshape(-1, *position_ids.size()[2:])
            target_ids = target_ids.reshape(-1, target_ids.size(-1))
            logit_mask = logit_mask.reshape(-1, logit_mask.size(-1))
            if prompt_pos is not None:
                prompt_pos = prompt_pos.reshape(-1, prompt_pos.size(-1))
        outputs, *mems = self.model(input_ids, position_ids, attention_mask, prompt_pos=prompt_pos)
        if self.take_softmax:
            outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # select the target logits
        batch_ids = torch.arange(target_ids.size(0), dtype=torch.long, device=target_ids.device)
        batch_ids = batch_ids.unsqueeze(1).expand_as(target_ids)
        seq_ids = torch.arange(target_ids.size(-1), dtype=torch.long, device=target_ids.device)
        seq_ids = seq_ids.unsqueeze(0).expand_as(target_ids)
        logits = outputs[batch_ids, seq_ids, target_ids]
        logits = (logits * logit_mask).sum(dim=1)
        if self.length_penalty > 0.0:
            logits = logits / logit_mask.sum(dim=1) ** self.length_penalty
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        return (logits, *mems)

class FP16_Module(torch.nn.Module):
    def __init__(self, module):
        super(FP16_Module, self).__init__()
        self.add_module('module', module.half())

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, FLOAT_TYPES):
            val = val.half()
        return val

    return conversion_helper(val, half_conversion)


def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, HALF_TYPES):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)