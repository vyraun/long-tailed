# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from typing import List, NamedTuple, Optional
# typing library provides support for type hints, type aliases, new types, simplifying complex signatures
# type checking verifies and enforces the constraints of the types, ensuring that the program is type-safe
from torch import Tensor

# So, this is an application: could have been done alternatively as:
# class EncoderOut(NamedTuple):
#   encoder_out: Tensor
#   src_lengths: Tensor
# Also, Optional is type hint for Union[..., None]

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor),  # B x T
        ("encoder_embedding", Tensor),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)

# This is the encoder template
class FairseqEncoder(nn.Module):
    """Base class for encoders."""
    
    # The constructor takes in a dictionary
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    # Forward takesn in source tokens and source lengths
    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        raise NotImplementedError

    # reorder the forward method's output
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        raise NotImplementedError

    # Okay, why is this here: 1 milion is the maximum length
    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 1e6  # an arbitrary large number

    # Don't know what this does: why is this required?
    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict
