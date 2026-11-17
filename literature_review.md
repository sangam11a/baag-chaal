# Literature Review & Critical Analysis: AI for Bagh-Chal and Related Games

> **Project**: AI Agents for Bagh-Chal — Q1 Journal Paper (Springer Nature)  
> **Date**: 2026-11-16  
> **Status**: Initial survey complete. Papers identified: 12 primary, ~20 secondary.

---

## Table of Contents
1. [Papers Directly on Bagh-Chal](#1-papers-directly-on-bagh-chal)
2. [Source Paper: Dhumbal (Malla, 2026)](#2-source-paper-dhumbal-malla-2026)
3. [Asymmetric Game AI](#3-asymmetric-game-ai)
4. [AlphaZero / MuZero Lineage](#4-alphazero--muzero-lineage)
5. [MCTS and Search Methods](#5-mcts-and-search-methods)
6. [Reinforcement Learning for Board Games](#6-reinforcement-learning-for-board-games)
7. [Graph Neural Networks for Games](#7-graph-neural-networks-for-games)
8. [Gap Analysis & Positioning](#8-gap-analysis--positioning)

---

## 1. Papers Directly on Bagh-Chal

### 1.1 Lim & Nievergelt (2009) — "Tigers and Goats is a Draw"

| Field | Detail |
|---|---|
| **Full Title** | Computing Tigers and Goats / Tigers and Goats is a Draw |
| **Authors** | Yew Jin Lim, Jürg Nievergelt |
| **Venue** | ICGA Journal (approx. 2009) |
| **Type** | Game solution via retrograde analysis |

**Summary**: Definitively proved that Bagh-Chal is a **draw** under perfect play from both sides. Used retrograde analysis to solve the sliding (movement) phase completely, producing an endgame database of **88,260,972 unique positions** (after removing symmetries). The placement phase (game tree complexity ~10^41) was solved using heuristic-optimal opponent confrontation and mid-tree databases to bridge the gap from the opening to the solved endgame.

**Critical Review**:

| Dimension | Score | Analysis |
|---|---|---|
| **Novelty** | 9/10 | First and only complete solution of a non-trivial asymmetric traditional board game. Landmark result. |
| **Rigor** | 10/10 | Exhaustive computation — every position in the sliding phase is resolved. The placement phase solution is verified end-to-end. |
| **Reproducibility** | 5/10 | The endgame database (88M positions) is not publicly available in a downloadable format. The computational methodology is described but code is not released. |
| **Relevance to Our Work** | **Critical** | This is our **ground truth oracle**. Any Bagh-Chal AI paper MUST reference this result. We can use the known game-theoretic value (draw) as the gold standard for evaluating agent quality. |

**Strengths**:
- Mathematically rigorous proof of game-theoretic value
- Established the state-space and game-tree complexity metrics that all subsequent work cites
- Clever two-phase approach (retrograde + forward search with mid-tree databases)

**Weaknesses / Gaps**:
- No evaluation of approximate AI agents (heuristic, search, RL)
- Endgame database not publicly accessible — limits use by subsequent researchers
- The paper focuses on the solution, not on understanding strategic patterns
- Published in a specialized game AI venue, not broadly cited in ML literature

**What We Take**: The draw result as our oracle. The complexity metrics (10^41 game tree, 8.83×10^7 endgame positions) for our paper's formalization section.

**What We Improve**: We provide approximate agents evaluated against the solved result, create a publicly accessible endgame database subset, and analyze strategic patterns (tiger forks, goat rings, etc.).

---

### 1.2 Raut et al. (2026) — "Deep Reinforcement Learning to Master the Asymmetric Strategy of Baghchal"

| Field | Detail |
|---|---|
| **Full Title** | Deep Reinforcement Learning to Master the Asymmetric Strategy of Baghchal |
| **Authors** | Ranjit Raut et al. |
| **Venue** | arXiv:2607.18296 (preprint, July 2026) |
| **Type** | Deep RL benchmark |

**Summary**: Systematically trained and evaluated four deep RL algorithms — **DQN, REINFORCE, PPO, and MuZero** — for both tiger and goat roles via self-play. Key results:
- **MuZero**: 86% win rate as tigers, 62% as goats (best overall)
- **PPO**: Competitive performance with significantly lower compute
- **DQN**: Heavy tiger bias due to dense immediate reward signals for captures
- **REINFORCE**: [Performance details not fully extracted — need to read paper]

**Critical Review**:

| Dimension | Score | Analysis |
|---|---|---|
| **Novelty** | 7/10 | First systematic deep RL benchmark for Bagh-Chal. Addresses a genuine gap. However, the algorithms (DQN, PPO, MuZero) are all off-the-shelf. No architectural innovation. |
| **Rigor** | 6/10 | [Preliminary — based on search summaries, not full paper read] Appears to test 4 algorithms but unclear on: number of seeds, statistical tests, effect sizes, confidence intervals. |
| **Baselines** | 5/10 | Missing: Minimax/Alpha-Beta (the natural baseline for perfect-info games), heuristic agents with domain knowledge, evaluation against the Lim-Nievergelt oracle. |
| **Evaluation** | 5/10 | Win rates reported but no Elo ratings, no side-specific breakdowns beyond aggregate, no role-balance analysis. |
| **Reproducibility** | 6/10 | arXiv preprint — code availability unclear. |
| **Statistical Validity** | 4/10 | [Need to verify] — likely no Cohen's d, no bootstrap, no Bonferroni correction. |
| **Relevance to Our Work** | **Very High — Threat** | This is our most direct competitor. Published as arXiv preprint in July 2026. If it reaches a Q1 journal before us, our novelty is severely diminished. |

**Strengths**:
- Covers 4 RL algorithms (more than most game-specific studies)
- Identifies the tiger-bias problem in DQN (important insight)
- MuZero result (86% tiger, 62% goat) provides a strong RL benchmark

**Weaknesses / Gaps We Can Exploit**:
1. **No search baselines**: No Minimax, Alpha-Beta, or MCTS-only baseline. Our paper includes these, providing a more complete picture.
2. **No heuristic agents**: No hand-coded expert agents. The Dhumbal paper showed heuristics can outperform RL — we test this hypothesis for Bagh-Chal.
3. **No architectural ablation**: Single-network only (presumably). Our dual-network vs. single-network comparison is novel.
4. **No representation comparison**: Likely uses flat MLP or basic CNN. Our GNN ablation is novel.
5. **No oracle evaluation**: Does not compare against the Lim-Nievergelt solved result. Our endgame DB evaluation is a differentiator.
6. **No statistical rigor**: Likely no effect sizes, CIs, or bootstrap. Our statistical suite (following Malla 2026) is stronger.
7. **Preprint only**: Not yet peer-reviewed. A well-executed journal paper has higher impact.

**What We Take**: MuZero and PPO as RL baselines to compare against. The tiger-bias observation as a known phenomenon to analyze.

**What We Improve**: Add search and heuristic baselines. Add architectural/representation ablations. Add oracle evaluation. Add rigorous statistics.

---

### 1.3 Thapa & Poudel (2024) — "AI Strategy Approach Development on Baghchal using AlphaZero"

| Field | Detail |
|---|---|
| **Full Title** | AI Strategy Approach Development on Baghchal using AlphaZero |
| **Authors** | Rajendra Bahadur Thapa, Laxmi Kanta Poudel |
| **Venue** | Journal of Science and Technology, Vol. 4 No. 2 (ICSET-2024), pp. 35–39 |
| **DOI** | 10.3126/jost.v4i2.78948 |
| **Type** | Conference paper (short) |

**Summary**: Compares heuristic evaluation methods against AlphaZero-style MCTS for Bagh-Chal. Evaluated over 20,000 games. Findings: heuristic bots are competitive but struggle with early recognition of deep win/lose trajectories. MCTS provides more robust strategy. Both sides' optimal play leads to draw (consistent with Lim & Nievergelt).

**Critical Review**:

| Dimension | Score | Analysis |
|---|---|---|
| **Novelty** | 4/10 | Applies existing AlphaZero methodology to Bagh-Chal. The heuristic vs. MCTS comparison is standard. |
| **Rigor** | 3/10 | Short conference paper (5 pages). 20,000 games but unclear evaluation methodology. No statistical tests mentioned. |
| **Depth** | 3/10 | Very brief. No ablations, no architectural details, no representation discussion. |
| **Reproducibility** | 2/10 | Published in a small Nepali journal. No code availability mentioned. |
| **Relevance to Our Work** | **Moderate** | Must be cited as prior work. Easy to surpass in rigor and depth. |

**Strengths**:
- Confirms the draw result independently (via simulation)
- Published in a peer-reviewed venue (even if small)

**Weaknesses / Gaps**:
- Extremely short — lacks methodological detail
- No RL comparison (only heuristic vs. MCTS)
- No statistical rigor
- Published in a venue with minimal visibility

**What We Take**: Citation as prior work showing heuristic vs. MCTS for Bagh-Chal.

**What We Improve**: Everything — depth, rigor, baselines, statistical analysis.

---

## 2. Source Paper: Dhumbal (Malla, 2026)

### 2.1 Malla (2026) — "Heuristic Optimization of AI Agents for Dhumbal"

| Field | Detail |
|---|---|
| **Full Title** | Heuristic optimization of AI agents for Dhumbal |
| **Authors** | Sahaj Raj Malla |
| **Venue** | Scientific Reports 16:33849 (2026) |
| **DOI** | 10.1038/s41598-026-68206-1 |
| **Type** | Full journal article (18 pages) |

**Summary**: Comprehensive AI benchmark for Dhumbal (Nepali card game, imperfect information). Compares rule-based (4 heuristic variants), search-based (MCTS, ISMCTS), RL (DQN, PPO), and random agents. Key result: Aggressive heuristic agent dominates with 88.9% win rate in cross-category championship. RL agents fail catastrophically (PPO at 0.2%, below Random baseline). Statistical analysis includes Cohen's d, 95% CI, Bonferroni correction, round-level bootstrap.

**Critical Review**: See Phase 1 of the main plan document for detailed scoring.

| Dimension | Score | Key Issue |
|---|---|---|
| **Novelty** | 5/10 | Benchmark paper, no new algorithm |
| **Rigor** | 6/10 | Good stats but RL catastrophically undertrained |
| **Baselines** | 4/10 | Missing CFR/MCCFR, NFSP |
| **Evaluation** | 6/10 | 2-player/4-player format mismatch for search agents |
| **Statistics** | 7/10 | Cohen's d + CI + bootstrap — strong |
| **Reproducibility** | 8/10 | Open source + Zenodo DOI |
| **Writing** | 7/10 | Clear, cautious claims |
| **Limitations Honesty** | 8/10 | Four explicit limitations + ethics section |

**Relevance to Our Work**: Methodological template. We replicate the statistical rigor and tournament structure while avoiding the fatal weaknesses (undertrained RL, single seed, format mismatch).

---

## 3. Asymmetric Game AI

### 3.1 Key References

| Paper | Game | Method | Key Finding | Relevance |
|---|---|---|---|---|
| Silver et al. (2017) — AlphaGo Zero | Go (symmetric) | Self-play RL + MCTS | Superhuman without human knowledge | Foundational. Bagh-Chal needs modifications for asymmetry. |
| Silver et al. (2018) — AlphaZero | Chess, Shogi, Go | Generalized AlphaGo Zero | Single algorithm masters 3 games | Template for our AlphaZero-style agent. |
| Schrittwieser et al. (2020) — MuZero | Atari, Go, Chess, Shogi | Model-based RL + MCTS (learned model) | Masters games without knowing the rules | Raut et al. used this; we should too. |
| Brown & Sandholm (2019) — Pluribus | Multiplayer Poker | MCCFR + real-time search | Superhuman in 6-player poker | Not directly applicable (imperfect info), but methodology reference. |

### 3.2 Asymmetric Games in AI Literature

| Game | Asymmetry Type | AI Work | Gap We Address |
|---|---|---|---|
| Fox & Hounds | 1 fox vs. 4 hounds, different objectives | Minimax solves it (simple) | Too simple for RL study |
| Scotland Yard | 1 vs. many, hidden movement | RL + belief states (limited work) | Imperfect info — different from Bagh-Chal |
| Hnefatafl/Tablut | King escape vs. attackers | AlphaZero adaptations (limited) | Closest analog to Bagh-Chal. No systematic Q1 paper. |
| Stratego | Hidden pieces, asymmetric info | DeepNash (2022) | Imperfect info |
| **Bagh-Chal** | **4 tigers (capture) vs. 20 goats (block)** | **This paper** | **Systematic comparison with oracle evaluation** |

**Critical Observation**: Hnefatafl is the closest structural analog to Bagh-Chal (asymmetric piece counts, predator-prey dynamics on a grid, perfect information). However, no Q1 journal paper systematically benchmarks RL architectures for Hnefatafl with the rigor we propose. This is our opportunity.

---

## 4. AlphaZero / MuZero Lineage

### Key Papers in the Lineage

| Year | Paper | Innovation | State-of-the-Art Status |
|---|---|---|---|
| 2016 | AlphaGo (Silver et al.) | CNN + MCTS + supervised learning from human games | Superhuman at Go |
| 2017 | AlphaGo Zero (Silver et al.) | Self-play only, no human data | Surpassed AlphaGo |
| 2018 | AlphaZero (Silver et al.) | Generalized to Chess, Shogi | Multi-game mastery |
| 2020 | MuZero (Schrittwieser et al.) | Learned dynamics model, no rules needed | Extended to Atari |
| 2021 | EfficientZero (Ye et al.) | Sample-efficient MuZero | 100× less data |
| 2022 | Sampled MuZero | Large discrete action spaces | Applicable to Bagh-Chal's variable action space |

**Critical Gap**: None of these papers address asymmetric games. All benchmark games (Go, Chess, Shogi, Atari) are either symmetric or single-player. Our architectural ablation (single vs. dual-network) for asymmetric AlphaZero fills this gap.

---

## 5. MCTS and Search Methods

### Relevant to Bagh-Chal

| Method | Paper | Application | Our Use |
|---|---|---|---|
| UCT (MCTS) | Browne et al. (2012) survey | General game-playing | Primary search baseline |
| ISMCTS | Cowling et al. (2012) | Imperfect-info card games | **NOT needed** (Bagh-Chal is perfect info) |
| Alpha-Beta | Standard (Knuth & Moore, 1975) | Perfect-info games | **Essential baseline** (missing from Raut et al.) |
| MCTS + Neural Network | AlphaZero | Go, Chess | Our AlphaZero-style agent |

**Critical Note**: Raut et al. (2026) did NOT include Minimax/Alpha-Beta in their comparison. This is a significant omission — for a **perfect-information** game, Alpha-Beta is the natural benchmark. Its absence means we cannot assess whether the RL agents' performance is actually impressive or merely comparable to a well-tuned search. **This is a key differentiator for our paper.**

---

## 6. Reinforcement Learning for Board Games

### Relevant Frameworks and Methods

| Framework/Method | Reference | Application | Notes |
|---|---|---|---|
| DQN | Mnih et al. (2015) | Atari games | Used by both Malla (2026) and Raut et al. (2026) |
| PPO | Schulman et al. (2017) | General RL | Best model-free method in both Dhumbal and Bagh-Chal |
| REINFORCE | Williams (1992) | Policy gradient | Used by Raut et al. — simplest policy gradient |
| A3C/A2C | Mnih et al. (2016) | Parallel training | Not yet applied to Bagh-Chal |
| SAC | Haarnoja et al. (2018) | Continuous action spaces | Not applicable (discrete actions) |
| RLCard | Zha et al. (2019) | Card game RL toolkit | Used in Dhumbal context |
| PyTAG | Balla et al. (2024) | Tabletop game RL | Potential framework for our implementation |

---

## 7. Graph Neural Networks for Games

### Motivation

Bagh-Chal's board is an **irregular graph**, not a regular grid:
- Corner nodes: 3 neighbors (2 orthogonal + 1 diagonal, or just 2 orthogonal for non-diagonal corners)
- Edge nodes: 3–5 neighbors
- Center node: 8 neighbors
- Other internal nodes: 4–8 neighbors depending on diagonal connectivity

A CNN treats the board as a 5×5 image, which **loses topological information** — it treats all positions as having 8 neighbors (with padding), ignoring that many diagonals don't exist.

### Relevant GNN Papers

| Paper | Application | Method | Relevance |
|---|---|---|---|
| Kipf & Welling (2017) | Graph classification | GCN | Foundational GNN architecture |
| Veličković et al. (2018) | Node classification | GAT (Graph Attention) | Attention-based — could weight important board positions |
| Battaglia et al. (2018) | Relational reasoning | Graph Networks | General framework for game state encoding |
| Cappart et al. (2021) | Combinatorial optimization | GNN + RL | Shows GNN + RL works for structured problems |

**Critical Gap**: No published work applies GNNs to Bagh-Chal or similar irregular board games for RL. This is a genuine novel contribution.

---

## 8. Gap Analysis & Positioning

### What Has Been Done

| Contribution | Lim (2009) | Thapa (2024) | Raut (2026) | Malla-Dhumbal (2026) |
|---|---|---|---|---|
| Game solution / oracle | ✅ (complete) | ❌ | ❌ | N/A (different game) |
| Heuristic agents | ❌ | ✅ (basic) | ❌ | ✅ (4 variants) |
| Minimax / Alpha-Beta | ❌ (used for solution) | ❌ | ❌ | N/A |
| MCTS | ❌ | ✅ (AlphaZero) | ✅ (MuZero) | ✅ (MCTS, ISMCTS) |
| DQN | ❌ | ❌ | ✅ | ✅ |
| PPO | ❌ | ❌ | ✅ | ✅ |
| MuZero | ❌ | ❌ | ✅ | ❌ |
| REINFORCE | ❌ | ❌ | ✅ | ❌ |
| GNN representation | ❌ | ❌ | ❌ | ❌ |
| Architecture ablation (single/dual-net) | ❌ | ❌ | ❌ | ❌ |
| Oracle-based evaluation | ❌ | ❌ | ❌ | ❌ |
| CMA-ES eval function tuning | ❌ | ❌ | ❌ | ❌ |
| Cohen's d / bootstrap statistics | ❌ | ❌ | ❌ (likely not) | ✅ |
| Multi-seed evaluation | ❌ | ❌ | ❌ (unclear) | ❌ (single seed) |
| Open-source + Zenodo | ❌ | ❌ | unclear | ✅ |

### Our Novel Contributions (Revised After Literature Survey)

| # | Contribution | Novelty Source | Risk Level |
|---|---|---|---|
| **C1** | **GNN representation** for Bagh-Chal's irregular board topology, compared against CNN and MLP | No prior work applies GNN to this game class | Low — GNN is well-established, application is novel |
| **C2** | **Dual-network vs. single-network architecture** ablation for asymmetric RL | Not done by Raut et al. or any Bagh-Chal paper | Medium — result may show no significant difference |
| **C3** | **Oracle-based evaluation** using Lim-Nievergelt endgame database as ground truth | Not done by any Bagh-Chal AI paper | Medium — requires database reconstruction |
| **C4** | **Most comprehensive agent comparison**: heuristic + Minimax/Alpha-Beta + MCTS + PPO + DQN + MuZero + AlphaZero-style | Raut et al. lack search baselines; Thapa lacks RL | Low — straightforward but labor-intensive |
| **C5** | **Statistical rigor**: Cohen's d, bootstrap, Bonferroni, multi-seed (following Malla 2026 standard) | Not present in Raut et al. or Thapa | Low — established methods |
| **C6** | **CMA-ES automated tuning** of Minimax evaluation function | Not done for Bagh-Chal | Low — CMA-ES is standard |

### Critical Risk: Raut et al. (2026)

The Raut et al. arXiv preprint is the biggest threat to our novelty. However:

1. It is **not yet peer-reviewed** (arXiv only as of July 2026)
2. It **lacks search baselines** (no Minimax, no Alpha-Beta)
3. It **lacks heuristic agents** (no domain-expert baselines)
4. It **lacks architectural ablation** (no single vs. dual network comparison)
5. It **lacks GNN representation**
6. It **lacks oracle evaluation** (no comparison to solved game value)
7. It **likely lacks statistical rigor** (no Cohen's d, bootstrap)

**Our paper is differentiated on at least 6 axes.** However, we must:
- Cite Raut et al. explicitly and acknowledge their prior work
- Frame our contribution as extending their RL-only benchmark with search, heuristic, and oracle comparisons
- Emphasize the methodological contributions (GNN, dual-network, oracle eval) over the "first to apply RL to Bagh-Chal" claim (which belongs to them)

---

## Summary Table: All Identified Papers

| # | Authors | Year | Title | Venue | Key Contribution | Our Relation |
|---|---|---|---|---|---|---|
| 1 | Lim & Nievergelt | 2009 | Tigers and Goats is a Draw | ICGA J. | Complete solution, endgame DB | Oracle reference |
| 2 | Thapa & Poudel | 2024 | AI Strategy on Baghchal using AlphaZero | J. Sci. Tech. Nepal | Heuristic vs MCTS | Prior work (low rigor) |
| 3 | Raut et al. | 2026 | Deep RL for Asymmetric Baghchal | arXiv | DQN/REINFORCE/PPO/MuZero comparison | Direct competitor |
| 4 | Malla | 2026 | Heuristic Optimization for Dhumbal | Sci. Reports | Comprehensive agent benchmark | Methodological template |
| 5 | Silver et al. | 2017 | AlphaGo Zero | Nature | Self-play RL + MCTS | Algorithm basis |
| 6 | Silver et al. | 2018 | AlphaZero | Science | Generalized self-play | Algorithm basis |
| 7 | Schrittwieser et al. | 2020 | MuZero | Nature | Learned model RL | Agent to implement |
| 8 | Browne et al. | 2012 | MCTS Survey | IEEE TCIAIG | MCTS foundations | Search baseline ref |
| 9 | Schulman et al. | 2017 | PPO | arXiv | Policy gradient method | Agent to implement |
| 10 | Mnih et al. | 2015 | DQN | Nature | Value-based RL | Agent to implement |
| 11 | Kipf & Welling | 2017 | GCN | ICLR | Graph convolution | Representation basis |
| 12 | Veličković et al. | 2018 | GAT | ICLR | Graph attention | Representation option |

---

> **Next Step**: Begin environment implementation. Read Raut et al. full paper when available to identify specific methodological gaps.

