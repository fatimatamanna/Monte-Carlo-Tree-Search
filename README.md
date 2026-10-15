# Monte Carlo Tree Search (MCTS)

**CSE 3500 Group Project** | Group #9

## Team Members

- Fatima Tamanna
- Jayson You
- Luka Zorin
- Kira Pyvovar

## About the Project

Monte Carlo Tree Search (MCTS) is a heuristic search algorithm and an AI algorithm used to find the best move in complex games by combining random simulations with an evolving tree structure. It is used in games such as Chess, Poker, Scrabble, and Go. Rather than calculating every possible combination of moves, MCTS estimates the value of a move by running repeated random simulations and uses those results to guide the search, balancing exploration of new options against exploitation of known good ones.

Our project researches, implements, and analyzes MCTS, and compares it to classical search algorithms such as minimax.

## How MCTS Works

MCTS repeats four phases until its time or iteration budget runs out:

1. **Selection:** Starting at the root, walk down the tree by choosing child nodes with the UCT (Upper Confidence bounds applied to Trees) formula.
2. **Expansion:** Add one or more new child nodes to the tree.
3. **Simulation (Rollout):** Play out a random sequence of moves from the new node to a terminal state.
4. **Backpropagation:** Pass the result back up the path, updating visit counts and win statistics.

UCT selects the child that maximizes:

```
w_i / n_i + C * sqrt( ln(N) / n_i )
```

where `w_i` is the number of wins for child `i`, `n_i` is its visit count, `N` is the parent's visit count, and `C` is the exploration constant.

## Objectives

- Understand the four phases of MCTS (Selection, Expansion, Simulation, Backpropagation) and how the UCT formula is used during Selection
- Understand how MCTS compares to other search algorithms (e.g., minimax) and what advantages it offers
- Implement the algorithm and test it on a game or decision-making problem
- Analyze performance and time complexity, and how the algorithm can be made more efficient
- Learn how to effectively research and implement MCTS as a group

## Project Timeline

| Week | Dates | Tasks |
|------|-------|-------|
| 4 | 9/21 - 9/25 | Research the algorithm and gather a basic understanding |
| 5 | 9/28 - 10/2 | Research related algorithms (e.g., minimax); compare time complexities |
| 6 | 10/5 - 10/9 | Decide on implementation approach; start implementation |
| 7 | 10/12 - 10/16 | Continue implementation |
| 8 | 10/19 - 10/23 | Continue implementation |
| 9 | 10/26 - 10/30 | Finalize implementation |
| 10 | 11/2 - 11/6 | Begin report and presentation slideshow |
| 11 | 11/9 - 11/13 | Continue report and slideshow |
| 12 | 11/16 - 11/20 | Finish and submit report; continue slideshow |
| Break | 11/23 - 11/27 | No work scheduled |
| 13 | 11/30 - 12/4 | Finalize presentation |
| 14 | 12/7 - 12/11 | Presentation |

## Repository Structure

> Update this section as the project takes shape.

```
.
├── README.md
├── src/        # MCTS implementation
├── tests/      # Test cases
├── report/     # Final report
└── slides/     # Presentation slides
```

## Getting Started

> Fill in once the implementation language and setup are decided.

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Build / run instructions go here
```

## Deliverables

- Working MCTS implementation with test cases
- Final report (introduction, comparison with other algorithms, detailed algorithm explanation, implementation details, results and analysis, time complexity analysis)
- Final presentation slides

## References

- Browne, C. et al. *A Survey of Monte Carlo Tree Search Methods.* IEEE Transactions on Computational Intelligence and AI in Games, 2012.
- Kocsis, L. and Szepesvári, C. *Bandit Based Monte-Carlo Planning.* ECML, 2006.