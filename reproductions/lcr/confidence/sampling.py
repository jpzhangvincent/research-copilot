"""
confidence/sampling.py

Multinomial sampling of K candidate answers from a black-box LLM (phi), used
as the first stage of the MSCP confidence metric (paper Sec 3.1, Eq. 1).

Two prompt templates are supported:
  - Query-only:       used to estimate C(q)    = MSCP(q; phi, K)
  - Query + document: used to estimate C(q, d) = MSCP(q, d; phi, K)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


QUERY_ONLY_PROMPT = (
    "Answer the following question as briefly as possible.\n"
    "Question: {query}\n"
    "Answer:"
)

QUERY_WITH_DOC_PROMPT = (
    "Answer the following question as briefly as possible.\n"
    "Context: {document}\n"
    "Question: {query}\n"
    "Answer:"
)


def build_sampling_prompt(query: str, document: Optional[str] = None) -> str:
    """Build the sampling prompt for either query-only or query+doc mode."""
    if document is None:
        return QUERY_ONLY_PROMPT.format(query=query)
    return QUERY_WITH_DOC_PROMPT.format(query=query, document=document)


class LLMClient(Protocol):
    """Minimal interface any backend (vLLM, HF transformers, API) must satisfy."""

    def generate(
        self,
        prompts: List[str],
        *,
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 64,
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Return, for each prompt, a list of `n` sampled completions."""
        ...


@dataclass
class SamplingConfig:
    K: int = 10
    temperature: float = 1.0
    max_tokens: int = 64
    stop: Optional[List[str]] = field(default_factory=lambda: ["\n"])


def sample_answers(
    llm: LLMClient,
    query: str,
    document: Optional[str] = None,
    config: Optional[SamplingConfig] = None,
) -> List[str]:
    """
    Draw K i.i.d. samples t_1..t_K from phi(.|x) with multinomial sampling,
    where x = query (if document is None) or (query, document).
    """
    config = config or SamplingConfig()
    prompt = build_sampling_prompt(query, document)
    completions = llm.generate(
        [prompt],
        n=config.K,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stop=config.stop,
    )[0]
    return [c.strip() for c in completions]


def sample_answers_batch(
    llm: LLMClient,
    queries: List[str],
    documents: Optional[List[Optional[str]]] = None,
    config: Optional[SamplingConfig] = None,
) -> List[List[str]]:
    """
    Batched version of sample_answers: builds one prompt per (query, document)
    pair and issues a single generate() call so vLLM can batch across the
    query-only + all top-10 documents for a test query (~11*K generations per
    query for MSCP alone, per the environment_setup throughput notes).
    """
    config = config or SamplingConfig()
    if documents is None:
        documents = [None] * len(queries)
    if len(documents) != len(queries):
        raise ValueError("queries and documents must have the same length")

    prompts = [build_sampling_prompt(q, d) for q, d in zip(queries, documents)]
    completions = llm.generate(
        prompts,
        n=config.K,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stop=config.stop,
    )
    return [[c.strip() for c in group] for group in completions]


class VLLMClient:
    """vLLM-backed LLMClient for Qwen2.5-7B-Instruct (recommended backend)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", **engine_kwargs):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vllm is required for VLLMClient; install with `pip install vllm`"
            ) from e
        self._SamplingParams = SamplingParams
        self._llm = LLM(model=model_name, **engine_kwargs)
        self._model_name = model_name

    def _apply_chat_template(self, prompt: str) -> str:
        tokenizer = self._llm.get_tokenizer()
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompts: List[str],
        *,
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 64,
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        chat_prompts = [self._apply_chat_template(p) for p in prompts]
        params = self._SamplingParams(
            n=n,
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            stop=stop,
        )
        outputs = self._llm.generate(chat_prompts, params)
        return [[o.text for o in out.outputs] for out in outputs]


class HFTransformersClient:
    """Fallback LLMClient using transformers, for environments without vLLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype)
        ).to(device)
        self._device = device

    def _apply_chat_template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompts: List[str],
        *,
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 64,
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        results: List[List[str]] = []
        for prompt in prompts:
            chat_prompt = self._apply_chat_template(prompt)
            inputs = self._tokenizer(chat_prompt, return_tensors="pt").to(self._device)
            gen = self._model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                num_return_sequences=n,
                max_new_tokens=max_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            texts = self._tokenizer.batch_decode(
                gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            if stop:
                trimmed = []
                for t in texts:
                    for s in stop:
                        idx = t.find(s)
                        if idx != -1:
                            t = t[:idx]
                    trimmed.append(t)
                texts = trimmed
            results.append(texts)
        return results
