# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import os
import re

# from socket import AF_ALG
from typing import Optional

from tokenizers import Regex
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

__all__ = [
    "Tokenizer",
    "DEFAULT_VOCAB",
]


class Tokenizer(PreTrainedTokenizerFast):
    """
    A tokenizer class mirroring the ESM-2 tokenizer's vocabulary and style, but without the <null_1> token.
    This results in a vocabulary size of 32 (a `multiple of 8`_) instead of ESM-2's 33.

    Parameters
    ----------
    vocab_file: str, optional
        Path to the vocabulary file. If not provided, the default vocabulary is used.
    bos_token: str, optional
        Beginning of sequence token. Default is "<cls>".
    eos_token: str, optional
        End of sequence token. Default is "<eos>".
    unk_token: str, optional
        Unknown token. Default is "<unk>".
    pad_token: str, optional
        Padding token. Default is "<pad>".
    mask_token: str, optional
        Mask token. Default is "<mask>".

    Any additional keyword arguments are passed to the `PreTrainedTokenizerFast` constructor.

    Notes
    -----
    .. _multiple of 8:
        https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/#:~:text=Simply%20padding%20the%20vocabulary%20size,performance%20of%20the%20projection%20layer.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        bos_token: str = "<cls>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        # load pretrained tokenizer (used by AutoTokenizer)
        if tokenizer_file is not None:
            super().__init__(
                tokenizer_file=tokenizer_file,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                pad_token=pad_token,
                mask_token=mask_token,
                **kwargs,
            )
            return

        # parse vocab
        if vocab_file is not None and os.path.isfile(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f if line.strip()]
        else:
            vocab = DEFAULT_VOCAB

        vocab_dict = {token: i for i, token in enumerate(vocab)}

        # create tokenizer
        tokenizer = HFTokenizer(
            WordLevel(
                vocab=vocab_dict,
                unk_token=unk_token,
            )
        )

        # special token regex
        special_start_char = Regex(r"[<\[]")
        special_end_char = Regex(r"[>\]]")

        # non-special token regex
        pattern = "|".join(re.escape(tok) for tok in vocab if len(tok) == 1)

        # pre-tokenization
        tokenizer.pre_tokenizer = Sequence(
            [
                Split(special_start_char, behavior="merged_with_next"),
                Split(special_end_char, behavior="merged_with_previous"),
                Split(Regex(pattern), behavior="isolated"),
            ]
        )

        # post-processing (add bos and eos tokens)
        tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            pair=f"{bos_token} $A $B {eos_token}",
            special_tokens=[
                (bos_token, vocab_dict[bos_token]),
                (eos_token, vocab_dict[eos_token]),
            ],
        )

        # initialize PreTrainedTokenizerFast
        super().__init__(
            tokenizer_object=tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )


DEFAULT_VOCAB = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "O",
    "Z",
    ".",
    "-",
    "<mask>",
]
