#!/usr/bin/env python3
"""
poker_mcts.py -- Monte Carlo Tree Search for heads-up fixed-limit Texas Hold'em
===============================================================================
CSE 3500, Group 9: Fatima Tamanna, Jayson You, Luka Zorin, Kira Pyvovar

Pure Python 3 (standard library only). Tested with Python 3.13.

    python poker_mcts.py                     # watch MCTS play itself, 5 hands
    python poker_mcts.py --hands 3 --iterations 2000 --c 1.0 --seed 7
    python poker_mcts.py --verbose           # also show the search statistics
    python poker_mcts.py --benchmark         # O(N) scaling table
    python poker_mcts.py --selftest          # correctness tests

-------------------------------------------------------------------------------
1. THE GAME (kept small so the betting tree stays bounded)
-------------------------------------------------------------------------------
* 2 players, 52-card deck, 2 hole cards each, 5 board cards.
* Each player antes 1 chip, so the pot starts at 2.
* Four streets: preflop, flop (3 cards), turn (1), river (1).
  Fixed bet sizes per street: 2, 2, 4, 4.
* Actions: FOLD, CALL (= check when nothing is owed), RAISE (= bet when
  nothing is owed). Every bet or raise is exactly the street's fixed size.
* At most 3 bets/raises per street. Player 0 acts first on every street.
* FOLD is only offered when facing a bet (folding for free is never useful),
  so the branching factor is b <= 3.

Longest possible street: check, bet, raise, raise, call = 5 actions, so a
whole hand is at most d = 20 decisions. The most a player can put in is
1 + 3*(2+2+4+4) = 37 chips, so dividing chip results by 37 maps every
payoff into [-1, 1].

-------------------------------------------------------------------------------
2. THE ALGORITHM: single-observer information-set MCTS (SO-ISMCTS)
-------------------------------------------------------------------------------
Each decision runs N iterations of the four classic MCTS phases:

  1. SELECTION       From the root, while the node is fully expanded, move to
                     the child maximizing the UCT score
                          w/n + C * sqrt( ln(N) / n )
                     w = total reward of the child, from the point of view of
                     the player who chose that move; n = child visits;
                     N = parent visits; C = exploration constant (--c).
  2. EXPANSION       If the node still has untried actions, try one at random
                     and add it to the tree as a new child.
  3. SIMULATION      Finish the hand with a fast randomized "rollout" policy
                     (probabilities stored in a lookup table).
  4. BACKPROPAGATION Walk back up the path. Each node adds the reward of the
                     player who MOVED INTO it, so selection at a parent always
                     maximizes the reward of the player acting there. For two
                     players this equals negamax; for more players it is the
                     max^n rule (used in poker_multiplayer.py).

Rewards are chip results divided by 37, so they lie in [-1, 1]. The move
actually played is the MOST-VISITED root child: visit counts are less noisy
than average values, which matters when the cards are random.

Hidden information. Poker hides the opponent's cards and future board cards,
but MCTS needs a complete state to simulate. So every iteration starts with
DETERMINIZATION: sample one complete deal consistent with what the deciding
player can see (their own two cards and the board cards already shown), then
run the iteration on that deal. All the samples share ONE tree whose edges
are betting actions only. Betting is public, so a node stands for what the
decider knows (the betting history), and its statistics average over many
possible hidden deals. That is the "single-observer" part.

The legal actions at a node depend only on the public betting history
(amount owed, raises so far), never on hidden cards. Every determinization
therefore sees the same legal moves at the same node, so plain UCT works
without the "availability count" correction that general ISMCTS needs.

Three refinements make the determinized search play sensibly:
  * BELIEF SAMPLING. The opponent's hand is not drawn uniformly. Each
    possible hand is weighted by how likely it was to produce the
    opponent's bets so far (Bayes' rule with a simple "stronger hands bet
    more" model; see BeliefSampler). Without this, a big re-raise is treated
    as if it could come from any two cards and the agent calls far too
    often. Every hand keeps a non-zero weight, so every sample is still a
    deal consistent with what the decider can see.
  * OPPONENT SPLITTING. One shared tree suffers from STRATEGY FUSION: a node
    averages over hidden cards, so the decision stored in it cannot depend
    on cards the decider would actually know. At an opponent's decision
    node, each iteration therefore descends into one of 5 sub-nodes chosen
    by the strength bucket of the opponent's SAMPLED hand. The tree learns
    that the opponent bets strong hands and folds weak ones, instead of
    modeling one card-blind "average opponent". (The deciding player's own
    hand is fixed, so its nodes need no split.)
  * TREE HORIZON. The tree only grows through the CURRENT betting round;
    later streets are played out by rollouts. A deeper tree would store our
    own turn and river decisions in nodes that average over every possible
    turn and river card, a blind decision we would never really make. With
    this horizon, extra iterations go into the decision we actually face,
    and play keeps improving as N grows.

-------------------------------------------------------------------------------
3. TIME AND SPACE COMPLEXITY
-------------------------------------------------------------------------------
Let b = branching factor (b <= 3), d = max hand length (d <= 20 actions),
N = iterations per decision, k = hidden cards to sample (k <= 7),
H = possible opponent hands (H <= C(50,2) = 1225).

  Determinize      copy the state once; draw the opponent's hand by
                   binary search on the belief; partial Fisher-Yates
                   for the board                                 O(k + log H)
  Selection        <= d levels, scan <= b children at each;
                   opponent split = one array lookup             O(b*d)
  Expansion        one new node                                  O(1)
  Simulation       <= d rollout moves, each a table lookup       O(d)
  Evaluation       two 7-card hand evaluations, single pass each O(7)
  Backpropagation  <= d nodes on the path                        O(d)
  -------------------------------------------------------------------------
  One iteration                                                  O(b*d)

Because b, d, k, and H are bounded constants for this game, one iteration
costs O(1) work and ONE DECISION COSTS O(N). Each iteration adds at most two
nodes (one new child plus one opponent-split sub-node), so MEMORY IS O(N).
Because of the tree horizon, the tree is also capped by the size T of one
betting round's tree, so memory is O(min(N, T)).

Before the iterations start, each decision does a fixed amount of setup:
the unseen-card pool, O(52), and the belief, which sorts the H possible
opponent hands on at most 4 streets, O(H log H). That cost does not depend
on N. `--benchmark` reports setup and search time separately; doubling N
roughly doubles the search time (ratio ~2.0x per row).

Why not compute the exact answer with full-width minimax/expectimax?
  * Deals: before the flop, player 0 does not know the opponent's 2 cards
    (C(50,2) = 1,225 options) or the 5 board cards (C(48,5) = 1,712,304).
    That is about 2.1 * 10^9 hidden-card deals.
  * Betting: this game's betting tree (bet cap 3) has 4,801 terminal
    betting sequences (counted by count_betting_tree() in the self-test).
  * Full-width search over every deal and every betting line would touch
    roughly 2.1e9 * 4.8e3, about 10^13 leaves, each needing a hand
    evaluation. Even at 10^8 leaves per second in C that is more than a day
    per decision; in Python (about 10^6 per second) it is months.
  * On top of that, minimax over determinized deals is not even correct for
    hidden-information games (strategy fusion again: it assumes you can play
    differently in situations you cannot tell apart).
MCTS is an anytime algorithm: it spends its budget of N samples on the lines
that look most promising and gives a usable answer for any N.

-------------------------------------------------------------------------------
4. CONSTANT-FACTOR OPTIMIZATIONS
-------------------------------------------------------------------------------
* The state is copied ONCE per iteration (inside determinize()); every move
  after that is applied IN PLACE. No per-node state copies.
* Cards are ints 0..51: rank = c >> 2, suit = c & 3. No objects, no strings.
* The 9 cards of a hand live in one flat list (p0, p0, p1, p1, 5 board),
  so copying is a single list slice.
* Lookup tables (built once): LEGAL[facing][can_raise] gives the legal
  action tuple, ROLLOUT_CDF[...] gives rollout probabilities, and
  STRAIGHT_HIGH[mask], TOP5[mask], POPCOUNT[mask] cover all 2^13 rank
  bitmasks.
* evaluate() is a single pass over the 7 cards. It builds "rank seen
  1/2/3/4 times" bitmasks plus per-suit bitmasks, then classifies the hand
  with a few bit operations and table lookups. It never generates all 21
  five-card subsets.
* Hidden cards come from a PARTIAL Fisher-Yates shuffle of the unseen-card
  pool: only k swaps for k cards. The pool is built once per decision and
  shuffled in place. A partial Fisher-Yates on ANY permutation of the pool
  gives a uniformly random k-subset, so the pool never needs resetting.
* Hot loops bind attributes and functions to local variables.
"""

import argparse
from bisect import bisect_left as _bisect_left, bisect_right as _bisect_right
import math
import random
import statistics
import sys
import time

# =============================================================================
# Cards
# =============================================================================
# A card is an int 0..51. rank = card >> 2 (0 = Two ... 12 = Ace),
# suit = card & 3 (0 = Clubs, 1 = Diamonds, 2 = Hearts, 3 = Spades).

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_WORDS = ("Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
              "Nine", "Ten", "Jack", "Queen", "King", "Ace")
RANK_PLURAL = ("Twos", "Threes", "Fours", "Fives", "Sixes", "Sevens", "Eights",
               "Nines", "Tens", "Jacks", "Queens", "Kings", "Aces")
SUIT_WORDS = ("Clubs", "Diamonds", "Hearts", "Spades")


def make_card(text):
    """'As' -> 51, 'Tc' -> 32. Used by the tests."""
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


def cards_from(text):
    """'As Kd 2c' -> [51, 46, 0]."""
    return [make_card(t) for t in text.split()]


def card_str(c):
    """Short form such as 'Ah'."""
    return RANKS[c >> 2] + SUITS[c & 3]


def card_words(c):
    """Beginner-friendly form such as 'Ace of Hearts'."""
    return RANK_WORDS[c >> 2] + " of " + SUIT_WORDS[c & 3]


# =============================================================================
# Hand evaluator (single pass, lookup tables)
# =============================================================================
# A hand's strength is one int. Bigger is better, equal means a tie.
#   bits 20..23 : category 0..8 (high card ... straight flush)
#   bits 16..19, 12..15, 8..11, 4..7, 0..3 : ranks that break ties, in order
# Comparing two ints therefore compares category first, then kickers.

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, \
    STRAIGHT_FLUSH = range(9)
HAND_NAMES = ("High card", "Pair", "Two pair", "Three of a kind", "Straight",
              "Flush", "Full house", "Four of a kind", "Straight flush")
_WHEEL = (1 << 12) | 0b1111          # A-2-3-4-5 ("the wheel")


def _build_tables():
    """Precompute per-13-bit-rank-mask tables (8192 entries each)."""
    straight_high = [-1] * 8192      # highest straight's top rank, or -1
    top5 = [0] * 8192                # top 5 ranks packed in nibbles 16..0
    popcount = [0] * 8192
    for m in range(8192):
        popcount[m] = bin(m).count("1")
        for hi in range(12, 3, -1):
            run = 0b11111 << (hi - 4)
            if m & run == run:
                straight_high[m] = hi
                break
        else:
            if m & _WHEEL == _WHEEL:
                straight_high[m] = 3     # the wheel is a Five-high straight
        packed, shift = 0, 16
        for r in range(12, -1, -1):
            if (m >> r) & 1:
                packed |= r << shift
                shift -= 4
                if shift < 0:
                    break
        top5[m] = packed
    return straight_high, top5, popcount


STRAIGHT_HIGH, TOP5, POPCOUNT = _build_tables()


def evaluate(cards):
    """Score the best 5-card poker hand among `cards` (normally 7 cards).

    Single pass over the cards, then O(1) bit operations. Works for 2..7
    cards; the UI uses that to name partial hands such as 'Pair' preflop.
    """
    m1 = m2 = m3 = m4 = 0            # ranks seen >=1, >=2, >=3, 4 times
    suit_masks = [0, 0, 0, 0]
    for c in cards:
        b = 1 << (c >> 2)
        if m1 & b:
            if m2 & b:
                if m3 & b:
                    m4 |= b
                else:
                    m3 |= b
            else:
                m2 |= b
        else:
            m1 |= b
        suit_masks[c & 3] |= b

    flush = 0
    for m in suit_masks:
        if POPCOUNT[m] >= 5:
            h = STRAIGHT_HIGH[m]
            if h >= 0:
                return (STRAIGHT_FLUSH << 20) | (h << 16)
            flush = m
            break

    if m4:                                           # four of a kind
        q = m4.bit_length() - 1
        return (QUADS << 20) | (q << 16) | ((TOP5[m1 & ~m4] >> 4) & 0xF000)
    if m3:                                           # full house?
        t = m3.bit_length() - 1
        other = m2 & ~(1 << t)          # another trips also counts as a pair
        if other:
            return (FULL_HOUSE << 20) | (t << 16) | ((other.bit_length() - 1) << 12)
    if flush:
        return (FLUSH << 20) | TOP5[flush]
    h = STRAIGHT_HIGH[m1]
    if h >= 0:
        return (STRAIGHT << 20) | (h << 16)
    if m3:
        t = m3.bit_length() - 1
        return (TRIPS << 20) | (t << 16) | ((TOP5[m1 & ~m3] >> 4) & 0xFF00)
    if m2:
        p = m2.bit_length() - 1
        rest = m2 & ~(1 << p)
        if rest:
            p2 = rest.bit_length() - 1
            kick = (TOP5[m1 & ~((1 << p) | (1 << p2))] >> 8) & 0x0F00
            return (TWO_PAIR << 20) | (p << 16) | (p2 << 12) | kick
        return (PAIR << 20) | (p << 16) | ((TOP5[m1 & ~(1 << p)] >> 4) & 0xFFF0)
    return TOP5[m1]                                  # high card


def hand_category(score):
    return score >> 20


def hand_name(score):
    """'Pair', 'Flush', ... ('Royal flush' for the Ace-high straight flush)."""
    cat = score >> 20
    if cat == STRAIGHT_FLUSH and ((score >> 16) & 15) == 12:
        return "Royal flush"
    return HAND_NAMES[cat]


def describe_hand(score):
    """Human-readable detail, e.g. 'Full house, Kings full of Fours'."""
    cat, r1, r2 = score >> 20, (score >> 16) & 15, (score >> 12) & 15
    if cat == HIGH_CARD:
        return "High card, " + RANK_WORDS[r1]
    if cat == PAIR:
        return "Pair of " + RANK_PLURAL[r1]
    if cat == TWO_PAIR:
        return "Two pair, %s and %s" % (RANK_PLURAL[r1], RANK_PLURAL[r2])
    if cat == TRIPS:
        return "Three of a kind, " + RANK_PLURAL[r1]
    if cat == STRAIGHT:
        return "Straight, %s-high" % RANK_WORDS[r1]
    if cat == FLUSH:
        return "Flush, %s-high" % RANK_WORDS[r1]
    if cat == FULL_HOUSE:
        return "Full house, %s full of %s" % (RANK_PLURAL[r1], RANK_PLURAL[r2])
    if cat == QUADS:
        return "Four of a kind, " + RANK_PLURAL[r1]
    if r1 == 12:
        return "Royal flush"
    return "Straight flush, %s-high" % RANK_WORDS[r1]


# =============================================================================
# Game constants and lookup tables
# =============================================================================
FOLD, CALL, RAISE = 0, 1, 2
ACTION_NAMES = ("FOLD", "CALL", "RAISE")
ANTE = 1
BET_SIZES = (2, 2, 4, 4)
MAX_RAISES = 3
MAX_CONTRIB = ANTE + MAX_RAISES * sum(BET_SIZES)       # 37 chips
STREET_NAMES = ("Preflop", "Flop", "Turn", "River", "Showdown")
VISIBLE_BOARD = (0, 3, 4, 5, 5)                        # board cards shown

# LEGAL[facing_bet][can_raise] -> tuple of legal actions
LEGAL = (
    ((CALL,), (CALL, RAISE)),                 # nothing to call: check / bet
    ((FOLD, CALL), (FOLD, CALL, RAISE)),      # facing a bet
)

# Rollout policy: cumulative probabilities, same indexing as LEGAL.
# A bit more passive than uniform random, which keeps rollouts realistic:
# real players rarely raise to the cap with nothing.
ROLLOUT_CDF = (
    (((1.0, CALL),), ((0.70, CALL), (1.0, RAISE))),
    (((0.25, FOLD), (1.0, CALL)), ((0.20, FOLD), (0.75, CALL), (1.0, RAISE))),
)


def partial_shuffle(pool, k, rnd):
    """Partial Fisher-Yates: after k swaps, pool[:k] is a uniform random
    k-subset of pool (in random order). O(k), in place.
    `rnd` is a bound random.random function (faster than randrange)."""
    n = len(pool)
    for i in range(k):
        j = i + int(rnd() * (n - i))
        pool[i], pool[j] = pool[j], pool[i]


# =============================================================================
# Opponent model: belief-weighted determinization
# =============================================================================
# Uniform determinization treats a big re-raise from the opponent as if it
# could come from any two cards. A real opponent's bets carry information, and
# a search that ignores it calls raises far too often.
#
# So we weight each possible opponent hand h by the likelihood of the actions
# the opponent actually took:
#       weight(h) = PRODUCT over opponent actions a_t of P(a_t | bucket_t(h))
# bucket_t(h) is h's hand-strength quintile (0 = weakest 20%, 4 = strongest
# 20%) among all hands the opponent could hold, measured on the board as it
# was when they acted. Every hand keeps a non-zero weight, so the sampled
# deals are still exactly the deals consistent with what we can see; they are
# just no longer equally likely. This is Bayes' rule with a simple model.
#
# Cost: once per decision, rank the H <= 1225 possible opponent hands on each
# street the opponent acted on (<= 4 sorts of H items), then each MCTS
# iteration draws a hand by binary search on cumulative weights, O(log H).
# H is a constant, so one decision is still O(N).
#
# The same buckets (on the current board) are used for OPPONENT SPLITTING in
# the tree: see MCTSAgent.search.

N_BUCKETS = 5
# ACTION_MODEL[bucket][facing_bet][can_raise] -> (P(fold), P(call), P(raise))
# Not a fitted model, just "stronger hands bet more and fold less", with
# some bluffing and slow-playing left in so no hand is ever ruled out.
ACTION_MODEL = (
    (((0.0, 1.0, 0.0), (0.0, 0.80, 0.20)), ((0.70, 0.30, 0.0), (0.70, 0.25, 0.05))),
    (((0.0, 1.0, 0.0), (0.0, 0.85, 0.15)), ((0.45, 0.55, 0.0), (0.45, 0.48, 0.07))),
    (((0.0, 1.0, 0.0), (0.0, 0.75, 0.25)), ((0.20, 0.80, 0.0), (0.20, 0.70, 0.10))),
    (((0.0, 1.0, 0.0), (0.0, 0.50, 0.50)), ((0.05, 0.95, 0.0), (0.05, 0.70, 0.25))),
    (((0.0, 1.0, 0.0), (0.0, 0.20, 0.80)), ((0.02, 0.98, 0.0), (0.02, 0.38, 0.60))),
)

_PREFLOP_EQUITY = None


def _preflop_key(a, b):
    """Starting-hand class: (high rank, low rank, suited). 169 classes."""
    ra, rb = a >> 2, b >> 2
    if ra < rb:
        ra, rb = rb, ra
    return (ra, rb, (a & 3) == (b & 3))


def preflop_equity(a, b):
    """Equity of a starting hand vs one random hand (169-class lookup table,
    computed once on first use by Monte Carlo with a fixed seed, ~0.3 s)."""
    global _PREFLOP_EQUITY
    if _PREFLOP_EQUITY is None:
        rng = random.Random(2024)
        rnd = rng.random
        table = {}
        for hi in range(13):
            for lo in range(hi + 1):
                for suited in ((False, True) if hi != lo else (False,)):
                    h = [hi * 4, lo * 4 + (0 if suited else 1)]
                    pool = [c for c in range(52) if c not in h]
                    score, samples = 0.0, 300
                    for _ in range(samples):
                        partial_shuffle(pool, 7, rnd)
                        board = pool[2:7]
                        x, y = evaluate(h + board), evaluate(pool[0:2] + board)
                        score += 1.0 if x > y else (0.5 if x == y else 0.0)
                    table[(hi, lo, suited)] = score / samples
        _PREFLOP_EQUITY = table
    return _PREFLOP_EQUITY[_preflop_key(a, b)]


def strength_percentiles(hands, board):
    """Percentile (0..1) of each 2-card hand among `hands` on `board`.
    Preflop uses the equity table; later streets use the made-hand score.
    O(H log H)."""
    if board:
        keys = [evaluate([a, b] + board) for a, b in hands]
    else:
        keys = [preflop_equity(a, b) for a, b in hands]
    order = sorted(keys)
    n2 = 2.0 * len(keys)
    bl, br = _bisect_left, _bisect_right
    return [(bl(order, k) + br(order, k)) / n2 for k in keys]


class BeliefSampler:
    """Weighted distribution over ONE opponent's two hidden cards.

    pool          : cards the observer cannot see
    visible_board : board cards the observer can see now
    opp_actions   : list of (street, facing, can_raise, action) the
                    opponent took this hand (public information)
    """

    def __init__(self, pool, visible_board, opp_actions, street_now=0,
                 need_buckets=False):
        n = len(pool)
        hands = [(pool[i], pool[j]) for i in range(n) for j in range(i + 1, n)]
        self.hands = hands
        self.cum = None                      # None means uniform
        self.bucket_now = None               # strength bucket on today's board
        by_street = {}
        for street, facing, can_raise, a in opp_actions:
            by_street.setdefault(street, []).append((facing, can_raise, a))
        if need_buckets:
            by_street.setdefault(street_now, [])
        if not by_street:
            return
        weights = [1.0] * len(hands)
        for street, acts in by_street.items():
            pct = strength_percentiles(hands, visible_board[:VISIBLE_BOARD[street]])
            buckets = [min(N_BUCKETS - 1, int(p * N_BUCKETS)) for p in pct]
            if street == street_now:
                self.bucket_now = buckets
            for facing, can_raise, a in acts:
                probs = [ACTION_MODEL[b][facing][can_raise][a] for b in range(N_BUCKETS)]
                for i, b in enumerate(buckets):
                    weights[i] *= probs[b]
        if not opp_actions:
            return
        cum, total = [], 0.0
        for w in weights:
            total += w
            cum.append(total)
        if total > 0.0:
            self.cum = cum

    def sample_index(self, u):
        """Index of one hand drawn with a uniform number u in [0,1). O(log H)."""
        cum = self.cum
        if cum is None:
            return int(u * len(self.hands))
        return _bisect_right(cum, u * cum[-1])

    def sample(self, u):
        return self.hands[self.sample_index(u)]

    def probability(self, i):
        """Probability the opponent holds hands[i] (for tests/explanations)."""
        if self.cum is None:
            return 1.0 / len(self.hands)
        prev = self.cum[i - 1] if i else 0.0
        return (self.cum[i] - prev) / self.cum[-1]


# =============================================================================
# Heads-up game state
# =============================================================================
class HUState:
    """Mutable heads-up hand. Moves are applied in place.

    cards   : flat list [p0a, p0b, p1a, p1b, b0, b1, b2, b3, b4]. The whole
              board is dealt up front; VISIBLE_BOARD[street] says how much is
              face up. Agents must only look at their own hole cards and the
              visible part of the board.
    contrib : chips each player has put in this hand (ante included)
    raises  : bets/raises made on the current street
    n_actions : actions taken on the current street
    """
    __slots__ = ("cards", "street", "to_act", "contrib", "raises",
                 "n_actions", "terminal", "folder", "_rewards", "history",
                 "pbucket")
    n_players = 2

    @classmethod
    def new_hand(cls, deck):
        s = cls.__new__(cls)
        s.cards = list(deck[:9])
        s.street = 0
        s.to_act = 0
        s.contrib = [ANTE, ANTE]
        s.raises = 0
        s.n_actions = 0
        s.terminal = False
        s.folder = -1
        s._rewards = None
        s.history = []       # public action log of the REAL hand only
        s.pbucket = None     # set on determinized copies: strength buckets
        return s

    def copy(self):
        s = HUState.__new__(HUState)
        s.cards = self.cards[:]
        s.street = self.street
        s.to_act = self.to_act
        s.contrib = self.contrib[:]
        s.raises = self.raises
        s.n_actions = self.n_actions
        s.terminal = self.terminal
        s.folder = self.folder
        s._rewards = None
        s.history = None     # simulations do not log (saves time)
        s.pbucket = self.pbucket
        return s

    # ---- views ------------------------------------------------------------
    def hole(self, p):
        return self.cards[2 * p:2 * p + 2]

    def visible_board(self):
        return self.cards[4:4 + VISIBLE_BOARD[self.street]]

    def pot(self):
        return self.contrib[0] + self.contrib[1]

    def to_call(self):
        p = self.to_act
        return self.contrib[1 - p] - self.contrib[p]

    def legal_actions(self):
        p = self.to_act
        c = self.contrib
        return LEGAL[c[1 - p] > c[p]][self.raises < MAX_RAISES]

    def rollout_action(self, u):
        """Pick a rollout move from the probability table using u in [0,1)."""
        p = self.to_act
        c = self.contrib
        for threshold, a in ROLLOUT_CDF[c[1 - p] > c[p]][self.raises < MAX_RAISES]:
            if u < threshold:
                return a
        return CALL

    # ---- transitions -------------------------------------------------------
    def apply(self, a):
        """Apply action a for the player to act, in place. O(1)."""
        p = self.to_act
        o = 1 - p
        c = self.contrib
        if self.history is not None:
            self.history.append((self.street, p, int(c[o] > c[p]),
                                 int(self.raises < MAX_RAISES), a))
        if a == FOLD:
            self.folder = p
            self.terminal = True
            return
        self.n_actions += 1
        if a == RAISE:
            c[p] = c[o] + BET_SIZES[self.street]
            self.raises += 1
            self.to_act = o
            return
        # CALL (or check). In heads-up the street closes once both players
        # have acted and the bets are equal, i.e. on any CALL that is not
        # the very first action of the street.
        c[p] = c[o]
        if self.n_actions >= 2:
            self.street += 1
            if self.street == 4:
                self.terminal = True
                return
            self.raises = 0
            self.n_actions = 0
            self.to_act = 0
        else:
            self.to_act = o

    # ---- results -----------------------------------------------------------
    def payoff0(self):
        """Net chips won by player 0 (player 1 gets the negative)."""
        c = self.contrib
        if self.folder == 0:
            return -c[0]
        if self.folder == 1:
            return c[1]
        cards = self.cards
        board = cards[4:9]
        s0 = evaluate(cards[0:2] + board)
        s1 = evaluate(cards[2:4] + board)
        if s0 > s1:
            return c[1]
        if s1 > s0:
            return -c[0]
        return 0                                   # split pot

    def payoffs(self):
        x = self.payoff0()
        return [x, -x]

    def rewards(self):
        """Payoffs scaled into [-1, 1] (cached)."""
        r = self._rewards
        if r is None:
            x = self.payoff0() / MAX_CONTRIB
            r = self._rewards = (x, -x)
        return r

    # ---- determinization ---------------------------------------------------
    def unseen_pool(self, observer):
        """Cards the observer cannot see. Built once per decision, O(52)."""
        seen = set(self.hole(observer))
        seen.update(self.visible_board())
        return [c for c in range(52) if c not in seen]

    def make_sampler(self, observer):
        """Everything determinize() needs, built ONCE per decision:
        the unseen-card pool plus a belief over the opponent's hand
        (including each candidate hand's strength bucket on today's board)."""
        pool = self.unseen_pool(observer)
        opp = 1 - observer
        acts = [(st, f, cr, a) for st, pl, f, cr, a in (self.history or []) if pl == opp]
        return pool, BeliefSampler(pool, self.visible_board(), acts,
                                   self.street, need_buckets=True)

    def determinize(self, observer, sampler, rnd):
        """Return a COPY of the state with the opponent's hole cards drawn
        from the belief and the unseen board cards drawn uniformly from the
        rest of the pool. The observer's cards and the visible board stay
        as they are. This is the single state copy made per MCTS iteration.
        O(k + log H), k <= 7 hidden cards, H <= 1225 possible opponent hands."""
        pool, belief = sampler
        s = self.copy()
        cards = s.cards
        vis = VISIBLE_BOARD[self.street]
        i = belief.sample_index(rnd())
        a, b = belief.hands[i]
        if belief.bucket_now is not None:
            s.pbucket = (belief.bucket_now[i],) * 2   # only the opponent's is read
        o = 2 * (1 - observer)
        cards[o] = a
        cards[o + 1] = b
        # Board: partial Fisher-Yates over the pool, skipping the (at most
        # two) cards just given to the opponent. need+2 draws always suffice.
        need = 5 - vis
        if need:
            partial_shuffle(pool, need + 2, rnd)
            j = 4 + vis
            for c in pool:
                if c != a and c != b:
                    cards[j] = c
                    j += 1
                    if j == 9:
                        break
        return s


# =============================================================================
# MCTS (generic: works for any state with the HUState interface, including
# the N-player state in poker_multiplayer.py)
# =============================================================================
class Node:
    """One tree node = one public betting history (an information set).

    mover : player who made the move INTO this node. value accumulates that
            player's reward, so a parent picking the child with the best
            value/visits picks the best move for whoever acts at the parent.
    """
    __slots__ = ("action", "mover", "visits", "value", "children", "untried",
                 "splits")

    def __init__(self, action, mover, legal):
        self.action = action
        self.mover = mover
        self.visits = 0
        self.value = 0.0
        self.children = []
        self.untried = list(legal)
        self.splits = None


TREE_HORIZON = 0     # betting rounds beyond the current one the tree may enter


class MCTSAgent:
    """Information-set MCTS with UCT selection.

    iterations : N, number of determinized iterations per decision
    c          : exploration constant C in  w/n + C*sqrt(ln N / n)
    """

    def __init__(self, iterations=500, c=1.0, rng=None, name=None):
        self.iterations = iterations
        self.c = c
        self.rng = rng or random.Random()
        self.name = name or "MCTS(%d)" % iterations
        self.last_root = None

    def act(self, state):
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        root = self.search(state, state.to_act)
        self.last_root = root
        # Final move: most-visited child (robust child).
        return max(root.children, key=lambda ch: ch.visits).action

    def search(self, state, observer):
        rnd = self.rng.random
        C = self.c
        log = math.log
        sqrt = math.sqrt
        last_street = state.street + TREE_HORIZON     # tree stays in this round
        t0 = time.perf_counter()
        sampler = state.make_sampler(observer)        # once per decision
        self.last_setup_seconds = time.perf_counter() - t0
        root = Node(None, observer, state.legal_actions())

        for _ in range(self.iterations):
            # Determinize: sample hidden cards, ONE state copy per iteration.
            s = state.determinize(observer, sampler, rnd)
            node = root
            path = [root]

            # 1. SELECTION: descend while the node is fully expanded and
            #    not terminal (terminal nodes have no children and no untried).
            while True:
                if not s.terminal and s.to_act != observer:
                    # Opponent decision: go to the sub-node for the
                    # opponent's strength bucket in THIS determinization.
                    sp = node.splits
                    if sp is None:
                        sp = node.splits = [None] * N_BUCKETS
                    b = s.pbucket[s.to_act]
                    sub = sp[b]
                    if sub is None:
                        sub = sp[b] = Node(None, node.mover, s.legal_actions())
                    node = sub
                    path.append(node)
                if node.untried or not node.children:
                    break
                k = C * sqrt(log(node.visits))
                best = None
                best_v = -1e300
                for ch in node.children:              # b <= 3 children
                    n = ch.visits
                    v = ch.value / n + k / sqrt(n)    # = w/n + C*sqrt(lnN/n)
                    if v > best_v:
                        best_v = v
                        best = ch
                node = best
                s.apply(node.action)
                path.append(node)

            # 2. EXPANSION: add one untried action as a new child, but only
            #    inside the tree horizon (the current betting round).
            untried = node.untried
            if untried and s.street <= last_street:
                i = int(rnd() * len(untried))
                a = untried[i]
                untried[i] = untried[-1]              # O(1) removal
                untried.pop()
                mover = s.to_act
                s.apply(a)
                child = Node(a, mover, () if s.terminal else s.legal_actions())
                node.children.append(child)
                path.append(child)

            # 3. SIMULATION: random rollout from the lookup-table policy.
            while not s.terminal:
                s.apply(s.rollout_action(rnd()))

            # 4. BACKPROPAGATION: each node takes its mover's reward.
            r = s.rewards()
            for nd in path:
                nd.visits += 1
                nd.value += r[nd.mover]
        return root


def tree_size(node):
    """Number of nodes in a search tree (used to show O(N) memory)."""
    total, stack = 0, [node]
    while stack:
        nd = stack.pop()
        total += 1
        stack.extend(nd.children)
        if nd.splits:
            stack.extend(x for x in nd.splits if x is not None)
    return total


# =============================================================================
# Self-play demo
# =============================================================================
def new_deck(rng):
    deck = list(range(52))
    rng.shuffle(deck)
    return deck


def show_search(agent):
    """Print the root statistics of the agent's last search."""
    root = agent.last_root
    if root is None:
        return
    for ch in sorted(root.children, key=lambda c: -c.visits):
        print("             %-5s visits %5d   avg reward %+.3f  (%+.2f chips)"
              % (ACTION_NAMES[ch.action], ch.visits, ch.value / ch.visits,
                 ch.value / ch.visits * MAX_CONTRIB))


def play_hand(agents, deck, verbose=False):
    """Play one heads-up hand between two MCTS agents and narrate it."""
    s = HUState.new_hand(deck)
    for p in (0, 1):
        print("  Player %d holds %s" % (p, " ".join(card_str(c) for c in s.hole(p))))
    street = -1
    while not s.terminal:
        if s.street != street:
            street = s.street
            board = s.visible_board()
            print("  -- %s%s" % (STREET_NAMES[street],
                  ("  board: " + " ".join(card_str(c) for c in board)) if board else ""))
        p = s.to_act
        agent = agents[p]
        agent.last_root = None
        a = agent.act(s)
        if a not in s.legal_actions():
            raise RuntimeError("MCTS chose an illegal action")
        s.apply(a)
        print("     Player %d %-5s  pot %d" % (p, ACTION_NAMES[a], s.pot()))
        if verbose:
            show_search(agent)
    pay = s.payoffs()
    if s.folder >= 0:
        print("  Player %d folds; player %d wins %+d" % (s.folder, 1 - s.folder, pay[1 - s.folder]))
    else:
        board = s.cards[4:9]
        print("  Showdown, board: " + " ".join(card_str(c) for c in board))
        for p in (0, 1):
            print("     Player %d: %s" % (p, describe_hand(evaluate(s.hole(p) + board))))
        print("  Result: player 0 %+d, player 1 %+d" % (pay[0], pay[1]))
    return pay


def self_play(hands, iterations, c, seed, verbose):
    rng = random.Random(seed)
    agents = [MCTSAgent(iterations, c, random.Random(seed + 1), "Player 0"),
              MCTSAgent(iterations, c, random.Random(seed + 2), "Player 1")]
    print("MCTS self-play: %d hand%s, N=%d iterations per decision, C=%.2f, seed %d"
          % (hands, "" if hands == 1 else "s", iterations, c, seed))
    total = [0, 0]
    t0 = time.perf_counter()
    for h in range(hands):
        print()
        print("Hand %d" % (h + 1))
        pay = play_hand(agents, new_deck(rng), verbose)
        total = [total[0] + pay[0], total[1] + pay[1]]
    elapsed = time.perf_counter() - t0
    print()
    print("Totals: player 0 %+d, player 1 %+d   (%.1f ms per hand)"
          % (total[0], total[1], 1000 * elapsed / max(1, hands)))


# =============================================================================
# Benchmark: one decision at increasing N, to show O(N) time and memory
# =============================================================================
def benchmark(c, seed, repeats=5):
    rng = random.Random(seed)
    state = HUState.new_hand(new_deck(rng))
    preflop_equity(0, 4)               # build the one-time lookup table first
    print("Benchmark: one preflop decision, median of %d runs, Python %s"
          % (repeats, sys.version.split()[0]))
    print("'search' excludes the fixed per-decision setup (unseen pool + belief),")
    print("which does not depend on N.")
    print("%10s %10s %10s %10s %8s %11s" % ("iterations", "setup ms", "search ms",
          "us/iter", "ratio", "tree nodes"))
    prev = None
    for n in (250, 500, 1000, 2000, 4000):
        times, setups = [], []
        for r in range(repeats):
            agent = MCTSAgent(n, c, random.Random(seed + r))
            t0 = time.perf_counter()
            root = agent.search(state, 0)
            total = time.perf_counter() - t0
            setups.append(agent.last_setup_seconds)
            times.append(total - agent.last_setup_seconds)
        t = statistics.median(times)
        ratio = "-" if prev is None else "%.2fx" % (t / prev)
        print("%10d %10.2f %10.1f %10.1f %8s %11d" % (
            n, 1000 * statistics.median(setups), 1000 * t, 1e6 * t / n, ratio,
            tree_size(root)))
        prev = t
    print("Doubling N roughly doubles the search time (ratio ~2.0x): O(N) per decision.")
    print("Each iteration adds at most 2 nodes, so memory is O(N); because the tree")
    print("only covers the current betting round it also levels off at that round's")
    print("size T, so memory is O(min(N, T)).")


# =============================================================================
# Self-test
# =============================================================================
def _reference_eval5(cards):
    """Slow, obviously correct 5-card evaluator, used only to check
    evaluate(). Produces the same packed score format."""
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    is_flush = len({c & 3 for c in cards}) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_hi = -1
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_hi = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_hi = 3

    def pack(cat, rs):
        v, shift = cat << 20, 16
        for r in rs:
            v |= r << shift
            shift -= 4
        return v

    if is_flush and straight_hi >= 0:
        return pack(STRAIGHT_FLUSH, [straight_hi])
    if groups[0][1] == 4:
        return pack(QUADS, [groups[0][0], groups[1][0]])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return pack(FULL_HOUSE, [groups[0][0], groups[1][0]])
    if is_flush:
        return pack(FLUSH, ranks)
    if straight_hi >= 0:
        return pack(STRAIGHT, [straight_hi])
    if groups[0][1] == 3:
        return pack(TRIPS, [g[0] for g in groups])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return pack(TWO_PAIR, [g[0] for g in groups])
    if groups[0][1] == 2:
        return pack(PAIR, [g[0] for g in groups])
    return pack(HIGH_CARD, ranks)


def count_betting_tree():
    """Count terminal betting sequences of one full hand (cards ignored)."""
    deck = list(range(9))
    leaves = 0
    stack = [HUState.new_hand(deck)]
    while stack:
        s = stack.pop()
        for a in s.legal_actions():
            t = s.copy()
            t.apply(a)
            if t.terminal:
                leaves += 1
            else:
                stack.append(t)
    return leaves


def selftest(seed=12345):
    from itertools import combinations
    rng = random.Random(seed)
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                               (" -- " + detail) if detail else ""))
        if not ok:
            failures += 1

    E = lambda text: evaluate(cards_from(text))
    print("Hand evaluator")
    ladder = [
        ("High card", "2c 5d 9h Js Kc 3d 7h"),
        ("Pair", "2c 2d 9h Js Kc 3d 7h"),
        ("Two pair", "2c 2d 9h 9s Kc 3d 7h"),
        ("Three of a kind", "2c 2d 2h 9s Kc 3d 7h"),
        ("Straight", "5c 6d 7h 8s 9c 2d Kh"),
        ("Flush", "2h 5h 9h Jh Kh 3d 7c"),
        ("Full house", "2c 2d 2h 9s 9c 3d 7h"),
        ("Four of a kind", "2c 2d 2h 2s Kc 3d 7h"),
        ("Straight flush", "5h 6h 7h 8h 9h 2d Kc"),
    ]
    scores = [E(t) for _, t in ladder]
    names_ok = all(hand_name(s) == n for s, (n, _) in zip(scores, ladder))
    check("hand-rank ordering (9 categories strictly increasing)",
          names_ok and all(a < b for a, b in zip(scores, scores[1:])),
          " < ".join(hand_name(s) for s in scores))
    wheel = E("Ac 2d 3h 4s 5c Kd 9h")
    six_high = E("2d 3h 4s 5c 6h Kd 9h")
    check("wheel A-2-3-4-5 is a Five-high straight",
          hand_name(wheel) == "Straight" and describe_hand(wheel) == "Straight, Five-high",
          describe_hand(wheel))
    check("wheel loses to a Six-high straight", wheel < six_high)
    check("ace-high straight beats king-high",
          E("Tc Jd Qh Ks Ac 2d 3h") > E("9c Tc Jd Qh Ks 2d 3h"))
    check("kickers: pair of Aces, King kicker > Queen kicker",
          E("Ac Ad Kh 9s 7c 4d 2h") > E("Ah As Qh 9s 7c 4d 2h"))
    check("kickers: 5th card counts for high card",
          E("Ac Kd 9h 7s 5c 3d 2h") > E("Ah Ks 9c 7d 4c 3h 2s"))
    check("kickers: only best 5 cards count (6th card ignored -> tie)",
          E("Ac Kd 9h 7s 5c 3d 2h") == E("Ah Ks 9c 7d 5h 4c 2s"))
    check("full house vs full house: Kings full of Twos > Queens full of Aces",
          E("Kc Kd Kh 2s 2c 5d 7h") > E("Qc Qd Qh As Ac 5d 7h"))
    check("two trips make the higher full house: Aces full of Kings",
          describe_hand(E("Ac Ad Ah Kc Kd Ks 2h")) == "Full house, Aces full of Kings")
    check("straight flush beats four of a kind",
          E("5h 6h 7h 8h 9h 9c 9d") > E("Ac Ad Ah As Kc 2d 3h"))
    check("royal flush is named", hand_name(E("Th Jh Qh Kh Ah 2c 3d")) == "Royal flush")

    mismatches = 0
    for _ in range(3000):
        seven = rng.sample(range(52), 7)
        ref = max(_reference_eval5(list(five)) for five in combinations(seven, 5))
        if evaluate(seven) != ref:
            mismatches += 1
    check("3000 random 7-card hands match brute-force best-of-21 evaluator",
          mismatches == 0, "%d mismatches" % mismatches)

    print("Game engine")
    leaves = count_betting_tree()
    check("betting tree is finite and bounded", leaves > 0, "%d terminal betting sequences" % leaves)
    bad_sum = bad_len = bad_contrib = 0
    max_len = 0
    for _ in range(2000):
        s = HUState.new_hand(new_deck(rng))
        steps = 0
        while not s.terminal:
            s.apply(rng.choice(s.legal_actions()))
            steps += 1
            if steps > 20:
                break
        max_len = max(max_len, steps)
        if not s.terminal or steps > 20:
            bad_len += 1
        if sum(s.payoffs()) != 0:
            bad_sum += 1
        if max(s.contrib) > MAX_CONTRIB:
            bad_contrib += 1
    check("2000 random hands terminate within 20 actions", bad_len == 0,
          "longest hand %d actions" % max_len)
    check("2000 random hands have zero-sum payoffs", bad_sum == 0)
    check("contributions never exceed %d chips" % MAX_CONTRIB, bad_contrib == 0)

    # Determinization keeps what the observer sees and never duplicates cards.
    ok = True
    for _ in range(300):
        s = HUState.new_hand(new_deck(rng))
        while not s.terminal and rng.random() < 0.8:
            s.apply(rng.choice(s.legal_actions()))
        if s.terminal:
            continue
        obs = s.to_act
        d = s.determinize(obs, s.make_sampler(obs), rng.random)
        if (d.hole(obs) != s.hole(obs) or d.visible_board() != s.visible_board()
                or len(set(d.cards)) != 9):
            ok = False
    check("determinization keeps the observer's view and has no duplicate cards", ok)

    print("MCTS")
    illegal = 0
    tested = 0
    for i in range(200):
        s = HUState.new_hand(new_deck(rng))
        while not s.terminal and rng.random() < 0.85:
            s.apply(rng.choice(s.legal_actions()))
        if s.terminal:
            continue
        agent = MCTSAgent(60, 0.5, random.Random(i))
        a = agent.act(s)
        tested += 1
        if a not in s.legal_actions():
            illegal += 1
    check("MCTS returns a legal action in %d random positions" % tested, illegal == 0)
    root = MCTSAgent(1000, 0.5, random.Random(1)).search(HUState.new_hand(new_deck(rng)), 0)
    check("root visits equal the iteration count", root.visits == 1000)
    check("tree has at most 2N+1 nodes (O(N) memory)", tree_size(root) <= 2001,
          "%d nodes" % tree_size(root))
    s = HUState.new_hand(cards_from("Ac Ad 2c 7d Ah As Kc Kd 3h"))   # seat 0: four Aces
    s.apply(CALL); s.apply(CALL)                                      # to the flop
    s.apply(CALL); s.apply(RAISE)                 # seat 0 checks, seat 1 bets
    root = MCTSAgent(2000, 0.5, random.Random(2)).search(s, 0)
    visits = {ACTION_NAMES[ch.action]: ch.visits for ch in root.children}
    best = max(root.children, key=lambda ch: ch.visits).action
    check("with four Aces facing a bet, MCTS keeps playing and rarely explores FOLD",
          best != FOLD and visits["FOLD"] == min(visits.values()), str(visits))

    print()
    print("SELFTEST %s (%d failure%s)" % ("PASSED" if failures == 0 else "FAILED",
                                          failures, "" if failures == 1 else "s"))
    return failures == 0


# =============================================================================
# Command line
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="MCTS agent for heads-up fixed-limit Texas Hold'em (CSE 3500 Group 9)")
    ap.add_argument("--hands", type=int, default=5,
                    help="self-play hands to show (default 5)")
    ap.add_argument("--iterations", type=int, default=500,
                    help="MCTS iterations per decision (default 500)")
    ap.add_argument("--c", type=float, default=1.0,
                    help="UCT exploration constant C (default 1.0)")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("--verbose", action="store_true",
                    help="show visit counts and values of each root action")
    ap.add_argument("--benchmark", action="store_true",
                    help="time one decision at 250..4000 iterations")
    ap.add_argument("--selftest", action="store_true", help="run correctness tests")
    args = ap.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    if args.selftest:
        return 0 if selftest() else 1
    if args.benchmark:
        benchmark(args.c, seed)
        return 0
    self_play(args.hands, args.iterations, args.c, seed, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
