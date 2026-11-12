import argparse
import math
import random
import time
from bisect import bisect_left, bisect_right
from itertools import accumulate, combinations

# ---------------------------------------------------------------------------
# Cards
# A card is an int 0-51: rank = card // 4 (0 = Two ... 12 = Ace), suit = card % 4
# ---------------------------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_NAMES = ["Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
              "Ten", "Jack", "Queen", "King", "Ace"]
SUIT_NAMES = ["Clubs", "Diamonds", "Hearts", "Spades"]
HAND_NAMES = ["High card", "Pair", "Two pair", "Three of a kind", "Straight",
              "Flush", "Full house", "Four of a kind", "Straight flush"]


def card_str(card):
    return RANKS[card // 4] + SUITS[card % 4]          # e.g. "Ah"


def card_name(card):
    return RANK_NAMES[card // 4] + " of " + SUIT_NAMES[card % 4]   # e.g. "Ace of Hearts"


def parse_cards(text):
    return [RANKS.index(t[0]) * 4 + SUITS.index(t[1]) for t in text.split()]


# ---------------------------------------------------------------------------
# Hand evaluator
# A hand's score is one int: category << 20, then up to 5 tie-breaking ranks
# in 4-bit slots. Comparing scores compares hands.
# ---------------------------------------------------------------------------

# Lookup tables over all 2^13 sets of ranks (built once):
#   STRAIGHT_HIGH[mask] = top rank of the best straight in mask, or -1
#   TOP5[mask]          = the 5 highest ranks in mask, packed into 4-bit slots
STRAIGHT_HIGH = [-1] * 8192
TOP5 = [0] * 8192
WHEEL = 0b1000000001111                                   # A-2-3-4-5
for mask in range(8192):
    for high in range(12, 3, -1):
        run = 0b11111 << (high - 4)
        if mask & run == run:
            STRAIGHT_HIGH[mask] = high
            break
    else:
        if mask & WHEEL == WHEEL:
            STRAIGHT_HIGH[mask] = 3                       # Five-high straight
    top = [r for r in range(12, -1, -1) if mask >> r & 1][:5]
    TOP5[mask] = sum(r << (16 - 4 * i) for i, r in enumerate(top))


def highest(mask):
    return mask.bit_length() - 1


def evaluate(cards):
    """Score the best 5-card hand among 2-7 cards in one pass over the cards."""
    seen = [0, 0, 0, 0]       # seen[k] = ranks that appear at least k+1 times
    suits = [0, 0, 0, 0]      # ranks held in each suit
    for c in cards:
        bit = 1 << (c // 4)
        k = 0
        while seen[k] & bit:
            k += 1
        seen[k] |= bit
        suits[c % 4] |= bit
    ranks, pairs, trips, quads = seen

    flush = next((m for m in suits if m.bit_count() >= 5), 0)
    if flush and STRAIGHT_HIGH[flush] >= 0:
        return 8 << 20 | STRAIGHT_HIGH[flush] << 16
    if quads:
        q = highest(quads)
        return 7 << 20 | q << 16 | (TOP5[ranks & ~quads] >> 4) & 0xF000
    if trips:
        t = highest(trips)
        other_pair = pairs & ~(1 << t)
        if other_pair:
            return 6 << 20 | t << 16 | highest(other_pair) << 12
    if flush:
        return 5 << 20 | TOP5[flush]
    if STRAIGHT_HIGH[ranks] >= 0:
        return 4 << 20 | STRAIGHT_HIGH[ranks] << 16
    if trips:
        return 3 << 20 | t << 16 | (TOP5[ranks & ~trips] >> 4) & 0xFF00
    if pairs:
        p = highest(pairs)
        rest = pairs & ~(1 << p)
        if rest:
            p2 = highest(rest)
            kicker = (TOP5[ranks & ~(1 << p | 1 << p2)] >> 8) & 0x0F00
            return 2 << 20 | p << 16 | p2 << 12 | kicker
        return 1 << 20 | p << 16 | (TOP5[ranks & ~(1 << p)] >> 4) & 0xFFF0
    return TOP5[ranks]


def hand_name(cards):
    return HAND_NAMES[evaluate(cards) >> 20]


# ---------------------------------------------------------------------------
# Game rules: fixed-limit hold'em for 2-6 players
# ---------------------------------------------------------------------------
FOLD, CALL, RAISE = 0, 1, 2
ACTION_NAMES = ["fold", "call", "raise"]
STREETS = ["Preflop", "Flop", "Turn", "River"]
BET_SIZES = [2, 2, 4, 4]
MAX_RAISES = 3
ANTE = 1
MAX_LOSS = ANTE + MAX_RAISES * sum(BET_SIZES)    # 37 chips, used to scale rewards
BOARD_SHOWN = [0, 3, 4, 5, 5]                     # face-up board cards per street

# Lookup tables indexed by [facing a bet][raise still allowed]
LEGAL = [[(CALL,), (CALL, RAISE)],
         [(FOLD, CALL), (FOLD, CALL, RAISE)]]
ROLLOUT = [[[(1.0, CALL)], [(0.7, CALL), (1.0, RAISE)]],     # cumulative probabilities
           [[(0.25, FOLD), (1.0, CALL)], [(0.2, FOLD), (0.75, CALL), (1.0, RAISE)]]]


class PokerState:
    """One hand. Moves are applied in place so MCTS only copies once per iteration.

    cards: player p holds cards[2p] and cards[2p+1]; the last 5 are the board.
    The whole board is dealt up front but only BOARD_SHOWN[street] cards are visible.
    """

    def __init__(self, n, deck, first=0):
        self.n = n
        self.cards = list(deck[:2 * n + 5])
        self.bets = [ANTE] * n          # chips each player has put in this hand
        self.folded = [False] * n
        self.first = first              # seat that acts first on every street
        self.to_act = first
        self.street = 0
        self.high_bet = ANTE
        self.raises = 0
        self.waiting = n                # players who still need to act this street
        self.done = False
        self.history = []               # (street, player, facing_bet, action)
        self.buckets = None             # opponents' sampled strength (MCTS only)

    def copy(self):
        s = PokerState.__new__(PokerState)
        s.__dict__.update(self.__dict__)
        s.cards = self.cards[:]
        s.bets = self.bets[:]
        s.folded = self.folded[:]
        s.history = None                # simulations don't need a log
        return s

    def hole(self, p):
        return self.cards[2 * p:2 * p + 2]

    def board(self):
        start = 2 * self.n
        return self.cards[start:start + BOARD_SHOWN[self.street]]

    def pot(self):
        return sum(self.bets)

    def to_call(self):
        return self.high_bet - self.bets[self.to_act]

    def legal_actions(self):
        return LEGAL[self.to_call() > 0][self.raises < MAX_RAISES]

    def rollout_action(self, rng):
        r = rng.random()
        for threshold, action in ROLLOUT[self.to_call() > 0][self.raises < MAX_RAISES]:
            if r < threshold:
                return action
        return CALL

    def next_player(self, p):
        p = (p + 1) % self.n
        while self.folded[p]:
            p = (p + 1) % self.n
        return p

    def apply(self, action):
        p = self.to_act
        if self.history is not None:
            self.history.append((self.street, p, int(self.to_call() > 0), action))
        if action == FOLD:
            self.folded[p] = True
            self.waiting -= 1
            if self.folded.count(False) == 1:
                self.done = True
                return
        elif action == CALL:
            self.bets[p] = self.high_bet
            self.waiting -= 1
        else:
            self.high_bet += BET_SIZES[self.street]
            self.bets[p] = self.high_bet
            self.raises += 1
            self.waiting = self.folded.count(False) - 1     # everyone else responds
        if self.waiting == 0:                               # betting round is over
            self.street += 1
            if self.street == 4:
                self.done = True
                return
            self.raises = 0
            self.waiting = self.folded.count(False)
            p = self.first - 1
        self.to_act = self.next_player(p)

    def winners(self):
        """Players who win the pot, in seat order starting from the first actor."""
        order = [(self.first + i) % self.n for i in range(self.n)]
        alive = [p for p in order if not self.folded[p]]
        if len(alive) == 1:
            return alive
        board = self.cards[2 * self.n:]
        scores = {p: evaluate(self.hole(p) + board) for p in alive}
        best = max(scores.values())
        return [p for p in alive if scores[p] == best]

    def payoffs(self):
        """Net chips per player. Ties split the pot; odd chips go to the first winners."""
        result = [-b for b in self.bets]
        winners = self.winners()
        share, extra = divmod(self.pot(), len(winners))
        for i, p in enumerate(winners):
            result[p] += share + (1 if i < extra else 0)
        return result

    def rewards(self):
        """Payoffs scaled into [-1, 1] for MCTS."""
        scale = MAX_LOSS * (self.n - 1)
        return [x / scale for x in self.payoffs()]


# ---------------------------------------------------------------------------
# Guessing hidden cards
# Each possible opponent hand is weighted by how likely it was to make the bets
# we saw (stronger hands bet more). Hands are ranked into 5 strength buckets.
# ---------------------------------------------------------------------------
BUCKETS = 5
# chance of (fold, call, raise) for each bucket: [not facing a bet, facing a bet]
ACTION_MODEL = [
    [(0, 0.80, 0.20), (0.70, 0.25, 0.05)],     # weakest 20% of hands
    [(0, 0.85, 0.15), (0.45, 0.48, 0.07)],
    [(0, 0.75, 0.25), (0.20, 0.70, 0.10)],
    [(0, 0.50, 0.50), (0.05, 0.70, 0.25)],
    [(0, 0.20, 0.80), (0.02, 0.38, 0.60)],     # strongest 20%
]


def preflop_score(a, b):
    """Rough starting-hand strength: high cards, pairs and suited cards are better."""
    hi, lo = max(a // 4, b // 4), min(a // 4, b // 4)
    score = 2 * hi + lo
    if hi == lo:
        score += 20
    if a % 4 == b % 4:
        score += 2
    return score


def strength_buckets(hands, board):
    """Bucket 0-4 for each hand: its strength percentile among all `hands`."""
    if board:
        keys = [evaluate(list(h) + board) for h in hands]
    else:
        keys = [preflop_score(a, b) for a, b in hands]
    order = sorted(keys)
    # percentile of each hand, counting ties as half below and half above
    return [BUCKETS * (bisect_left(order, k) + bisect_right(order, k)) // (2 * len(keys) + 1)
            for k in keys]


class Belief:
    """Weighted list of the two-card hands one opponent could be holding."""

    def __init__(self, hands, buckets_by_street, actions, street):
        weights = [1.0] * len(hands)
        for st, facing, action in actions:
            buckets = buckets_by_street[st]
            for i, b in enumerate(buckets):
                weights[i] *= ACTION_MODEL[b][facing][action]
        self.hands = hands
        self.buckets = buckets_by_street[street]
        self.cumulative = list(accumulate(weights))

    def sample(self, rng):
        i = bisect_right(self.cumulative, rng.random() * self.cumulative[-1])
        return self.hands[i], self.buckets[i]


def make_beliefs(state, me):
    """Done once per decision: the unseen cards and a Belief for each opponent."""
    seen = set(state.hole(me) + state.board())
    pool = [c for c in range(52) if c not in seen]
    hands = list(combinations(pool, 2))
    board = state.board()
    buckets_by_street = [strength_buckets(hands, board[:BOARD_SHOWN[st]])
                         for st in range(state.street + 1)]
    beliefs = []
    for p in range(state.n):
        if p != me:
            actions = [(st, f, a) for st, who, f, a in state.history if who == p]
            beliefs.append((p, Belief(hands, buckets_by_street, actions, state.street)))
    return pool, beliefs


def partial_shuffle(pool, k, rng):
    """Partial Fisher-Yates: shuffle only the first k positions, O(k)."""
    for i in range(k):
        j = rng.randrange(i, len(pool))
        pool[i], pool[j] = pool[j], pool[i]


def sample_deal(state, me, pool, beliefs, rng):
    """Copy the state and replace everything `me` can't see with a sampled guess."""
    s = state.copy()
    s.buckets = [0] * s.n
    used = set()
    for p, belief in beliefs:
        for _ in range(20):                      # redraw if a card is already taken
            (a, b), bucket = belief.sample(rng)
            if a not in used and b not in used:
                break
        else:
            a, b = rng.sample([c for c in pool if c not in used], 2)
        used.update((a, b))
        s.cards[2 * p], s.cards[2 * p + 1] = a, b
        s.buckets[p] = bucket
    shown = BOARD_SHOWN[s.street]
    need = 5 - shown
    partial_shuffle(pool, need + len(used), rng)
    fresh = [c for c in pool[:need + len(used)] if c not in used][:need]
    s.cards[2 * s.n + shown:] = fresh
    return s


# ---------------------------------------------------------------------------
# Monte Carlo Tree Search
# ---------------------------------------------------------------------------
class Node:
    def __init__(self, action, mover, actions):
        self.action = action        # action that led here
        self.mover = mover          # player who made that action
        self.visits = 0
        self.value = 0.0            # total reward for `mover`
        self.children = []
        self.untried = list(actions)
        self.split = None           # opponent nodes: one sub-node per strength bucket


class MCTSAgent:
    def __init__(self, iterations=500, c=1.0, seed=None):
        self.iterations = iterations
        self.c = c
        self.rng = random.Random(seed)
        self.last_root = None

    def act(self, state):
        actions = state.legal_actions()
        if len(actions) == 1:
            return actions[0]
        self.last_root = self.search(state)
        return max(self.last_root.children, key=lambda ch: ch.visits).action

    def uct(self, parent, child):
        return child.value / child.visits + self.c * math.sqrt(math.log(parent.visits) / child.visits)

    def search(self, state):
        me = state.to_act
        pool, beliefs = make_beliefs(state, me)
        root = Node(None, me, state.legal_actions())

        for _ in range(self.iterations):
            s = sample_deal(state, me, pool, beliefs, self.rng)
            node = root
            path = [root]

            # 1. Selection
            while True:
                if not s.done and s.to_act != me:
                    # Opponent's turn: use the sub-tree for how strong their sampled
                    # hand is, so the tree learns that strong hands bet more.
                    if node.split is None:
                        node.split = [None] * BUCKETS
                    b = s.buckets[s.to_act]
                    if node.split[b] is None:
                        node.split[b] = Node(None, node.mover, s.legal_actions())
                    node = node.split[b]
                    path.append(node)
                if node.untried or not node.children:
                    break
                node = max(node.children, key=lambda ch: self.uct(path[-1], ch))
                s.apply(node.action)
                path.append(node)

            # 2. Expansion (the tree only covers the current betting round)
            if node.untried and s.street == state.street:
                action = node.untried.pop(self.rng.randrange(len(node.untried)))
                mover = s.to_act
                s.apply(action)
                child = Node(action, mover, [] if s.done else s.legal_actions())
                node.children.append(child)
                path.append(child)

            # 3. Simulation
            while not s.done:
                s.apply(s.rollout_action(self.rng))

            # 4. Backpropagation: each node gets the reward of the player who moved into it
            rewards = s.rewards()
            for n in path:
                n.visits += 1
                n.value += rewards[n.mover]
        return root


# ---------------------------------------------------------------------------
# Demo and benchmark
# ---------------------------------------------------------------------------
def new_deck(rng):
    deck = list(range(52))
    rng.shuffle(deck)
    return deck


def show_cards(cards):
    return " ".join(card_str(c) for c in cards)


def demo(hands, iterations, c, seed, verbose):
    rng = random.Random(seed)
    bots = [MCTSAgent(iterations, c, seed + 1), MCTSAgent(iterations, c, seed + 2)]
    totals = [0, 0]
    print(f"MCTS vs MCTS, {iterations} iterations per decision, C = {c}")
    for h in range(hands):
        s = PokerState(2, new_deck(rng))
        print(f"\nHand {h + 1}: player 0 has {show_cards(s.hole(0))}, "
              f"player 1 has {show_cards(s.hole(1))}")
        street = -1
        while not s.done:
            if s.street != street:
                street = s.street
                print(f"  {STREETS[street]} {show_cards(s.board())}".rstrip())
            p = s.to_act
            action = bots[p].act(s)
            s.apply(action)
            print(f"    player {p} {ACTION_NAMES[action]:5}  pot {s.pot()}")
            if verbose and bots[p].last_root:
                for ch in bots[p].last_root.children:
                    print(f"        {ACTION_NAMES[ch.action]:5} visits {ch.visits:4}  "
                          f"avg reward {ch.value / ch.visits:+.3f}")
                bots[p].last_root = None
        result = s.payoffs()
        totals = [totals[0] + result[0], totals[1] + result[1]]
        if not any(s.folded):
            board = s.cards[4:]
            print(f"  Showdown: {hand_name(s.hole(0) + board)} vs {hand_name(s.hole(1) + board)}")
        print(f"  Result: player 0 {result[0]:+}, player 1 {result[1]:+}")
    print(f"\nTotals: player 0 {totals[0]:+}, player 1 {totals[1]:+}")


def benchmark(c, seed):
    state = PokerState(2, new_deck(random.Random(seed)))
    print("Time for one MCTS decision (best of 3)")
    print(f"{'iterations':>10} {'ms':>8} {'ratio':>7}")
    previous = None
    for n in (250, 500, 1000, 2000, 4000):
        times = []
        for run in range(3):
            start = time.perf_counter()
            MCTSAgent(n, c, seed + run).search(state)
            times.append(time.perf_counter() - start)
        t = min(times)
        ratio = f"{t / previous:.2f}x" if previous else "-"
        print(f"{n:>10} {1000 * t:>8.1f} {ratio:>7}")
        previous = t
    print("Doubling the iterations about doubles the time, so one decision is O(N).")


def main():
    parser = argparse.ArgumentParser(description="MCTS for heads-up limit Texas Hold'em")
    parser.add_argument("--hands", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--c", type=float, default=1.0, help="UCT exploration constant")
    parser.add_argument("--seed", type=int, default=random.randrange(10**6))
    parser.add_argument("--verbose", action="store_true", help="show MCTS visit counts")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        benchmark(args.c, args.seed)
    else:
        demo(args.hands, args.iterations, args.c, args.seed, args.verbose)


if __name__ == "__main__":
    main()