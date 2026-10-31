"""LLM reranking: a decoder-only large language model judges each (claim, abstract) pair.

The prompt gives the model an instruction, the claim and one abstract, and allows only
"yes" or "no" as the answer. Instead of generating text, we read the model's next-token
logits for "yes" and "no": the relevance score is their difference, the log-odds of
P(yes) (pointwise relevance judgement, as in monoT5 and Qwen3-Reranker; Zhang et al., 2025).
Like the cross-encoder, it reranks a first stage's top candidates via ``RerankRetriever``.

The same prompt runs on two LLMs of identical size and architecture:
  * Qwen3-0.6B: a general chat LLM, used zero-shot (prompting only);
  * Qwen3-Reranker-0.6B: the same LLM fine-tuned to answer exactly this question.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# SciFact task instruction used by the Qwen3 embedding/reranking models in their MTEB evaluation.
INSTRUCTION = "Given a scientific claim, retrieve documents that support or refute the claim"
_SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
           'Note that the answer can only be "yes" or "no".')
_PREFIX = f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"  # Qwen3 chat, thinking switched off


def format_pair(query: str, doc: str, instruction: str = INSTRUCTION) -> tuple[str, str, str]:
    """Prompt as (prefix, body, suffix); only the body may be truncated."""
    return _PREFIX, f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}", _SUFFIX


def pack_ids(prefix: list[int], body: list[int], suffix: list[int], max_length: int) -> list[int]:
    """Concatenate token ids, cutting the end of the body so the whole prompt fits in ``max_length``."""
    room = max_length - len(prefix) - len(suffix)
    if room < 1:
        raise ValueError(f"max_length={max_length} leaves no room for the document")
    return prefix + body[:room] + suffix


def yes_no_log_odds(last_logits: np.ndarray, yes_id: int, no_id: int) -> np.ndarray:
    """log P(yes)/P(no) from next-token logits (restricted to the two allowed answers)."""
    return last_logits[:, yes_id] - last_logits[:, no_id]


class LLMReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        instruction: str = INSTRUCTION,
        max_length: int = 512,
        batch_size: int = 8,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        prefix, _, suffix = format_pair("", "", instruction)
        self._prefix = self.tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix = self.tokenizer.encode(suffix, add_special_tokens=False)
        self._pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.no_id

    def _ids(self, query: str, text: str) -> list[int]:
        _, body, _ = format_pair(query, text, self.instruction)
        return pack_ids(self._prefix, self.tokenizer.encode(body, add_special_tokens=False), self._suffix,
                        self.max_length)

    @torch.inference_mode()
    def score(self, query: str, texts: list[str]) -> np.ndarray:
        prompts = [self._ids(query, t) for t in texts]
        order = np.argsort([len(p) for p in prompts], kind="stable")  # similar lengths per batch: less padding
        scores = np.empty(len(prompts))
        for start in range(0, len(order), self.batch_size):
            batch = [prompts[i] for i in order[start:start + self.batch_size]]
            width = max(len(p) for p in batch)
            # Left padding, so the last position of every row is the answer slot.
            ids = torch.tensor([[self._pad] * (width - len(p)) + p for p in batch], device=self.device)
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in batch], device=self.device)
            logits = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, :]
            scores[order[start:start + self.batch_size]] = yes_no_log_odds(
                logits.float().cpu().numpy(), self.yes_id, self.no_id)
        return scores

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
