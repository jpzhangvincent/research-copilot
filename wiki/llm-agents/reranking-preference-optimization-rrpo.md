# ReRanking Preference Optimization (RRPO)

## Summary
ReRanking Preference Optimization (RRPO) is a reinforcement learning framework that trains a Retrieval-Augmented Generation (RAG) reranker to optimize directly for downstream LLM generation quality, rather than for static, human-annotated relevance labels. It reframes reranking as a sequential decision-making process and uses feedback from a frozen LLM Reader as the reward signal, removing the need for expensive human annotation.

## Key Ideas
- Standard rerankers are trained on static IR metrics (e.g., NDCG) decoupled from the downstream LLM, creating a mismatch between "topical relevance" and actual "usefulness" for answer generation.
- RRPO formulates document reranking as a finite-horizon Markov Decision Process (MDP), where the reranker sequentially selects documents from a candidate set.
- The reward at each step comes directly from evaluating the LLM Reader's generated response against the ground-truth answer, aligning training with generation quality rather than relevance labels.
- A reference-anchored deterministic baseline stabilizes training against the instability of discrete, high-variance LLM rewards, avoiding the need for a separate critic network.
- The training objective borrows the PPO-clip mechanism plus a KL-divergence penalty against a fixed reference policy, mirroring techniques used in RLHF for language models.
- RRPO is presented as a plug-and-play module: it generalizes across LLM Readers (including proprietary models like GPT-4o), composes with query expansion methods (e.g., Query2Doc), and tolerates noisy supervisor feedback.

## Method
RRPO models reranking as an MDP $(\mathcal{S}, \mathcal{A}, P, R)$:
- **State** $s_t$: the set of candidate documents not yet selected, starting from the top-$N$ retrieved documents.
- **Action** $a_t$: selecting one document from the current state's remaining candidates.
- **Transition**: deterministic — the selected document is removed from the state to form the next state.
- **Reward** $r_t$: computed by feeding the documents selected so far, along with the query and an instruction, into a frozen LLM Reader to generate a response; an evaluation metric $R_{lm}$ scores this response against the ground-truth answer.

The policy is realized by a pointwise reranking model $f_\theta$ that scores each document $(q, d_i)$ and converts scores into probabilities via softmax. At each RL step, the probability of selecting a given document is that document's initial probability normalized over the probabilities of the remaining (unselected) documents. Training uses a PPO-style clipped objective with importance-sampling ratios $\rho_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{ref}(a_t|s_t)$, an estimated advantage $\hat{A}_t$, and a KL-divergence penalty against a fixed reference policy $\pi_{ref}$ — augmented with the paper's reference-anchored deterministic baseline to reduce reward variance and encourage prioritizing document "usefulness" over mere topical relevance.

## Results
The provided text states that RRPO significantly outperforms strong supervised baselines and surpasses the list-wise LLM reranker RankZephyr on knowledge-intensive benchmarks. It also generalizes to diverse LLM Readers (including GPT-4o), yields additive gains when combined with the Query2Doc query expansion method, and remains effective when trained with feedback from smaller, noisy supervisors. No specific numerical metrics (e.g., NDCG, EM, F1 scores) are included in the provided text.

## Why It Matters
By optimizing the reranker with feedback from the actual downstream LLM Reader instead of static relevance labels, RRPO directly targets the true objective of a RAG system — generation quality — rather than a proxy IR metric. This closes the gap between "relevant" and "useful" documents, and its plug-and-play, annotation-free design makes it practical to apply across different readers and retrieval pipelines without costly human labeling.

---
Sources: Yuhang Wu, Xiangqing Shen, Fanfan Wang, Cangqi Zhou, Zhen Wu, Xinyu Dai, Rui Xia — arXiv, 2 Jul 2026
Raw: https://arxiv.org/html/2604.02091
Updated: 2026-07-24
