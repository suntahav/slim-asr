import torch
from torch import Tensor

def get_attn_pad_mask(inputs, input_length, expand_length):
    
    def get_transformer_non_pad_mask(inputs, input_length):
        batch_Size = inputs.size(0)
        
        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size())  # B x T
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1])  # B x T
        else:
            raise ValueError(f"Unsupported input shape {inputs.size()}")
        
        for i in range(batch_Size):
            non_pad_mask[i, input_length[i] :] = 0
        
        return non_pad_mask
    
    non_pad_mask = get_transformer_non_pad_mask(inputs, input_length)
    pad_mask = non_pad_mask.lt(1)
    attn_pad_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_pad_mask

def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)

    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask