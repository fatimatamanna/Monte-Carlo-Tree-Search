# Monte Carlo Tree Search on Poker

**CSE 3500, Group 9:** Fatima Tamanna, Jayson You, Luka Zorin, Kira Pyvovar

Our semester project implements **Monte Carlo Tree Search (MCTS)** for
fixed-limit Texas Hold'em, with attention to time complexity: one decision
costs **O(N)** time and **O(N)** memory for N iterations.

- `poker_mcts.py`: the MCTS agent for heads-up (2-player) Hold'em, with the
  game engine, a single-pass 7-card hand evaluator, a self-play demo, a
  scaling benchmark and self-tests.
- `poker_multiplayer.py`: play against MCTS robots at a 2 to 6 player table
  (you plus 3 robots by default), with a beginner-friendly text interface.

Both files are pure Python 3 with **no external libraries**. They were tested
with Python 3.13.

## Repository structure

```
.
├── README.md              this file
├── poker_mcts.py          heads-up engine + MCTS agent (the core algorithm)
└── poker_multiplayer.py   N-player engine + human-vs-robots game
                           (imports poker_mcts.py, so keep both in one folder)
```

## How to run

Install Python 3.13 from python.org. Nothing else is needed. On Windows, open
Command Prompt or PowerShell in the project folder (use `py` instead of
`python` if `python` is not on your PATH).

### Heads-up MCTS (`poker_mcts.py`)

```
python poker_mcts.py                                # MCTS plays itself, 5 hands
python poker_mcts.py --hands 3 --iterations 2000 --seed 7
python poker_mcts.py --verbose                      # show visits/values per decision
python poker_mcts.py --benchmark                    # O(N) scaling table
python poker_mcts.py --selftest                     # correctness tests
```

| Flag | Meaning | Default |
|---|---|---|
| `--hands` | self-play hands to show | 5 |
| `--iterations` | MCTS iterations N per decision | 500 |
| `--c` | UCT exploration constant C | 1.0 |
| `--seed` | random seed (same seed = same game) | random |
| `--verbose` | print each root action's visit count and average reward | off |
| `--benchmark` | time one decision at N = 250, 500, 1000, 2000, 4000 | |
| `--selftest` | run the evaluator, engine and MCTS tests | |

### Play against the robots (`poker_multiplayer.py`)

```
python poker_multiplayer.py                         # you vs 3 robots, 10 hands
python poker_multiplayer.py --players 6 --hands 20 --iterations 800
python poker_multiplayer.py --benchmark             # robot decision timing
python poker_multiplayer.py --selftest
```

| Flag | Meaning | Default |
|---|---|---|
| `--hands` | hands to play | 10 |
| `--iterations` | robot MCTS iterations per decision | 500 |
| `--players` | total players including you (2 to 6) | 4 |
| `--c` | UCT exploration constant | 1.0 |
| `--seed` | random seed | random |
| `--benchmark` | time one robot decision at N = 250 ... 4000 | |
| `--selftest` | run the multiplayer tests | |

During the game, every one of your turns starts with a box that shows the
street, the pot, the amount to call, your cards and the table cards in words
("Nine of Clubs, Ten of Spades"), how many table cards are still coming, your
current hand type, and each player's chips in and whether they folded. Type
`f` (fold), `c` (check/call), `r` (bet/raise) or `h` (help: rules, commands
and hand rankings). Only legal options are offered. After each hand you see
the showdown, the hand result and running totals. Press Enter for the next
hand or `q` to quit; final totals are printed best to worst.

## The game

Fixed-limit Texas Hold'em, kept small so the betting tree stays bounded:

- 52-card deck, 2 hole cards per player, 5 board cards.
- Every player antes 1 chip. No blinds, all-ins or side pots; only net chips
  are tracked, so nobody busts.
- Four streets (preflop, flop, turn, river) with fixed bets of 2, 2, 4, 4.
- Actions: fold, call/check, raise/bet. At most 3 bets/raises per street.
- Heads-up: player 0 always acts first. Multiplayer: the first player to act
  rotates every hand. Ties split the pot (odd chips go to the winner nearest
  the first seat).

## How the MCTS works

Each decision runs N iterations of the four classic phases:

1. **Selection**: from the root, while a node is fully expanded, move to the
   child with the highest UCT score `w/n + C * sqrt(ln(N) / n)`.
2. **Expansion**: add one untried action as a new child.
3. **Simulation**: finish the hand with a fast random rollout whose action
   probabilities come from a lookup table.
4. **Backpropagation**: walk back up the path. Each node adds the reward of the
   player who moved into it, so selection always favors the player acting at
   that node (negamax for 2 players, the max^n rule for more). Rewards are
   chip results scaled into [-1, 1].

The move played is the **most-visited** root child.

**Hidden cards (information-set MCTS).** Poker hides the opponents' cards and
the future board, so each iteration first samples a full deal that is
consistent with what the deciding player can see (**determinization**). All
samples share **one tree** whose edges are public betting actions
(single-observer ISMCTS). Legal actions depend only on the public betting, so
plain UCT works at every node.

Three refinements make the search play sensibly:

- **Belief sampling.** Opponent hands are weighted by how likely they were to
  produce the opponent's bets so far (Bayes' rule with a "stronger hands bet
  more" model), so a big raise is not treated as if it could be any two
  cards.
- **Opponent splitting.** At an opponent's decision node, the iteration enters
  one of 5 sub-nodes chosen by the sampled hand's strength bucket. The tree
  learns that opponents bet strong hands and fold weak ones, which reduces
  *strategy fusion* (one node averaging over cards a player would actually
  know).
- **Tree horizon.** The tree only grows through the current betting round;
  later streets are covered by rollouts. A deeper tree would store our own
  future decisions in nodes that average over every possible turn and river
  card. With the horizon, extra iterations keep improving the decision we
  actually face.

## Time complexity

Let b ≤ 3 (actions), d ≤ 20 (actions in a hand), k ≤ 7 hidden cards to sample
(heads-up), and H ≤ 1225 possible opponent hands.

| Phase | Work per iteration |
|---|---|
| Determinize: one state copy, opponent hand by binary search on the belief, partial Fisher-Yates for the board | O(k + log H) |
| Selection: ≤ d levels, ≤ b children each, O(1) opponent split | O(b·d) |
| Expansion | O(1) |
| Simulation: ≤ d moves, each a table lookup | O(d) |
| Evaluation: two single-pass 7-card evaluations | O(7) |
| Backpropagation | O(d) |
| **One iteration** | **O(b·d)** |

Because b, d, k and H are bounded constants, one iteration is O(1) and **one
decision is O(N)**. Each iteration adds at most two nodes, so **memory is
O(N)**; with the tree horizon it is also capped by the size T of one betting
round's tree, i.e. O(min(N, T)). Each decision also does a fixed setup that
does not depend on N: build the unseen-card pool (O(52)) and the belief
(O(H log H)). With P players, k = 2(P−1) + 5 and showdowns evaluate P hands:
the player count is a constant factor, so a decision is still O(N).

**Why not full-width minimax?** Before the flop, a player cannot see the
opponent's 2 cards (C(50,2) = 1,225 options) or the 5 board cards
(C(48,5) = 1,712,304): about 2.1 × 10⁹ hidden deals. This game's betting tree
has 4,801 terminal betting sequences, so exhaustive expectimax would touch
roughly 10¹³ leaves per decision, each needing a hand evaluation. That is
more than a day per decision even in C, and months in Python. Minimax over
determinized deals is also unsound for hidden-information games (strategy
fusion). MCTS is anytime: it gives a usable answer for any budget N.

**Keeping constant factors low:**

- The state is copied once per iteration; moves are then applied in place.
- Cards are ints (rank = c >> 2, suit = c & 3) in one flat list.
- Lookup tables cover legal actions, rollout probabilities, straights and
  top-5 ranks for all 2¹³ rank masks.
- A single-pass 7-card evaluator uses bitmasks instead of checking all 21
  five-card subsets.
- A partial Fisher-Yates shuffle draws only the k hidden cards needed.

### Measured scaling

`python poker_mcts.py --benchmark` (one preflop decision, median of 5 runs,
Python 3.13, Linux). The "search" column excludes the fixed per-decision
setup:

| Iterations | Setup ms | Search ms | µs/iter | Ratio | Tree nodes |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.67 | 1.7 | 6.9 | – | 98 |
| 500 | 0.64 | 3.3 | 6.6 | 1.92× | 110 |
| 1000 | 0.62 | 6.6 | 6.6 | 1.98× | 113 |
| 2000 | 0.62 | 12.7 | 6.4 | 1.94× | 113 |
| 4000 | 0.66 | 26.5 | 6.6 | 2.08× | 113 |

Doubling N doubles the search time, so the cost is linear. The tree stops at
113 nodes once the preflop betting round is fully expanded (the O(min(N, T))
memory bound).

`python poker_multiplayer.py --benchmark` (4 players, robot deciding on the
flop):

| Iterations | Setup ms | Search ms | µs/iter | Ratio | Tree nodes |
|---:|---:|---:|---:|---:|---:|
| 250 | 5.98 | 5.1 | 20.2 | – | 397 |
| 500 | 6.37 | 11.1 | 22.1 | 2.19× | 798 |
| 1000 | 5.43 | 18.5 | 18.5 | 1.67× | 1648 |
| 2000 | 5.35 | 36.9 | 18.5 | 2.00× | 3150 |
| 4000 | 5.48 | 76.0 | 19.0 | 2.06× | 5463 |

With 6 players it is about 28 µs per iteration: the player count only
changes the constant factor. Timings vary a little from run to run and
machine to machine; the ratios are the point.

## Tests

`--selftest` in each file checks:

- **Evaluator:** ordering of all 9 hand categories, the wheel (A-2-3-4-5) as
  a Five-high straight, kickers, full house vs full house, straight flush vs
  four of a kind, and 3,000 random 7-card hands against a brute-force
  best-of-21 reference evaluator (0 mismatches).
- **Heads-up engine:** 2,000 random hands end within 20 actions with zero-sum
  payoffs; the determinization keeps what the player can see and never
  duplicates a card.
- **Multiplayer engine:** 2,000 random hands for each table size from 2 to 6
  players end with zero-sum payoffs; everyone calling down puts in equal
  chips; a fold-around gives the last player everyone's ante; split pots
  including the odd chip; first-actor rotation; the 2-player engine matches
  the heads-up engine on 1,000 random hands.
- **MCTS:** always returns a legal action (heads-up and 2 to 6 players), root
  visits equal N, the tree has at most 2N+1 nodes, and rewards stay in
  [-1, 1].
