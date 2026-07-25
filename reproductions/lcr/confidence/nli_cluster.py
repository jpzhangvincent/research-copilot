"""Pairwise NLI entailment prompting and semantic clustering for MSCP (paper Sec 3.1, Eq. 1).

Given K sampled answers to a query (optionally conditioned on a document), this module
prompts the same LLM used for sampling to judge pairwise semantic entailment between
answers, builds an undirected graph where an edge exists iff entailment holds in BOTH
directions, and returns the connected components (semantic equivalence classes) of that
graph. Both a brute-force O(K^2) implementation (for correctness checks) and an
incremental O(K*M) implementation (representative-comparison, used in practice) are
provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from confidence.sampling import LLMClient

ENTAILMENT = "entailment"
CONTRADICTION = "contradiction"
NEUTRAL = "neutral"
_VALID_LABELS = (ENTAILMENT, CONTRADICTION, NEUTRAL)

NLI_PROMPT_TEMPLATE = (
    'We are evaluating answers to the question "{query}"\n'
    "Possible Answer 1: {answer_1}\n"
    "Possible Answer 2: {answer_2}\n"
    "Does Possible Answer 1 semantically entail Possible Answer 2? "
    "Respond with only one of: entailment, contradiction, or neutral."
)


def build_nli_prompt(query: str, answer_1: str, answer_2: str) -> str:
    """Builds the ordered pairwise entailment prompt f_phi(answer_1, answer_2)."""
    return NLI_PROMPT_TEMPLATE.format(query=query, answer_1=answer_1, answer_2=answer_2)


def parse_entailment_label(response: str) -> str:
    """Normalizes a raw LLM completion into one of entailment/contradiction/neutral.

    Falls back to `neutral` if the model's response does not clearly contain one of
    the three expected labels (robust to extra whitespace/punctuation/case).
    """
    text = response.strip().lower()
    for label in _VALID_LABELS:
        if label in text:
            return label
    return NEUTRAL


@dataclass
class NLIConfig:
    """Decoding hyperparameters for the pairwise entailment judgment calls.

    Unlike the MSCP sampling calls (temperature=1, multinomial), entailment judgments
    should be near-deterministic, so temperature defaults to 0.
    """

    max_tokens: int = 8
    temperature: float = 0.0
    stop: Optional[List[str]] = None


def query_entailment(
    llm: LLMClient,
    query: str,
    answer_1: str,
    answer_2: str,
    config: Optional[NLIConfig] = None,
) -> str:
    """Issues a single ordered pairwise entailment query f_phi(answer_1, answer_2)."""
    config = config or NLIConfig()
    prompt = build_nli_prompt(query, answer_1, answer_2)
    outputs = llm.generate(
        [prompt],
        n=1,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stop=config.stop,
    )
    return parse_entailment_label(outputs[0][0])


def query_entailment_batch(
    llm: LLMClient,
    query: str,
    pairs: List[Tuple[str, str]],
    config: Optional[NLIConfig] = None,
) -> List[str]:
    """Batched ordered pairwise entailment queries, one prompt per (answer_1, answer_2) pair."""
    if not pairs:
        return []
    config = config or NLIConfig()
    prompts = [build_nli_prompt(query, a1, a2) for a1, a2 in pairs]
    outputs = llm.generate(
        prompts,
        n=1,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stop=config.stop,
    )
    return [parse_entailment_label(out[0]) for out in outputs]


def is_bidirectional_entailment(label_ij: str, label_ji: str) -> bool:
    """Edge condition: (t_i, t_j) in E iff f_phi(t_i,t_j)=f_phi(t_j,t_i)=entailment."""
    return label_ij == ENTAILMENT and label_ji == ENTAILMENT


class UnionFind:
    """Standard union-find with path halving, used to compute connected components."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

    def components(self) -> List[List[int]]:
        groups: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return list(groups.values())


def cluster_answers_bruteforce(
    llm: LLMClient,
    query: str,
    answers: List[str],
    config: Optional[NLIConfig] = None,
) -> List[List[int]]:
    """Brute-force O(K^2) pairwise-NLI clustering.

    Computes f_phi for every ordered pair (i != j), builds the bidirectional-entailment
    graph G=(V,E), and returns its connected components as lists of answer indices.
    Used to validate the optimized incremental clustering below.
    """
    n = len(answers)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    ordered_pairs: List[Tuple[int, int]] = [
        (i, j) for i in range(n) for j in range(n) if i != j
    ]
    batch_pairs = [(answers[i], answers[j]) for i, j in ordered_pairs]
    labels = query_entailment_batch(llm, query, batch_pairs, config)
    label_map = {pair: label for pair, label in zip(ordered_pairs, labels)}

    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if is_bidirectional_entailment(label_map[(i, j)], label_map[(j, i)]):
                uf.union(i, j)
    return uf.components()


def cluster_answers_incremental(
    llm: LLMClient,
    query: str,
    answers: List[str],
    config: Optional[NLIConfig] = None,
) -> List[List[int]]:
    """Incremental O(K*M) pairwise-NLI clustering (paper Sec 3.1 optimization).

    Processes answers in order; each new answer is compared bidirectionally against
    one representative per existing cluster (rather than every prior answer). It is
    assigned to the first cluster whose representative bidirectionally entails it,
    otherwise it seeds a new cluster. This reduces the number of NLI calls from
    O(K^2) to O(K*M), where M is the number of clusters discovered so far.

    Returns a list of clusters, each a list of indices into `answers`; the clusters
    partition range(len(answers)) (sum of cluster sizes == K == len(answers)).
    """
    n = len(answers)
    if n == 0:
        return []

    clusters: List[List[int]] = [[0]]
    representatives: List[int] = [0]

    for idx in range(1, n):
        assigned = False
        for cluster_pos, rep in enumerate(representatives):
            label_fwd, label_bwd = query_entailment_batch(
                llm,
                query,
                [(answers[idx], answers[rep]), (answers[rep], answers[idx])],
                config,
            )
            if is_bidirectional_entailment(label_fwd, label_bwd):
                clusters[cluster_pos].append(idx)
                assigned = True
                break
        if not assigned:
            clusters.append([idx])
            representatives.append(idx)

    return clusters
