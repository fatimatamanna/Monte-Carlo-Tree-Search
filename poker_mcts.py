"""
Monte Carlo Tree Search (MCTS) for heads-up Texas Hold'em Poker
CSE 3500 - Group 9

Game rules used (kept small and fixed so the game tree has bounded depth):
  * Heads-up (2 players), standard 52-card deck, 2 hole cards each, 5 board cards.
  * Each player antes 1 chip. Player 0 acts first on every street.
  * Streets: preflop, flop, turn, river. Fixed bet sizes per street: 2, 2, 4, 4.
  * Actions: FOLD, CALL (call or check), RAISE (bet or raise by the street's fixed size).
  * At most 3 raises per street.

Because opponent cards are hidden (imperfect information), each MCTS iteration
"determinizes" the game: it samples a random opponent hand and board runout that is
consistent with what the player can see, then runs the usual four MCTS phases on that
sampled world (single-observer information-set MCTS). One tree is shared by all
sampled worlds, keyed by the betting history.

Time complexity
---------------
Let N = number of iterations, b = branching factor (at most 3 actions) and
d = maximum number of betting actions in a hand (a small constant here, <= 20).

  Selection        O(b) per tree level  -> O(b * d)
  Expansion        O(1)
  Simulation       O(d) random actions + one O(1) 7-card hand evaluation
  Backpropagation  O(d)
  Determinization  O(1) (partial Fisher-Yates shuffle of at most 7 cards)

So one iteration costs O(b * d), and one move decision costs O(N * b * d). Since b and d
are constants in this game, that is O(N), linear in the number of iterations, and
memory is O(N) nodes. A full-width minimax search would also have to enumerate every
possible hidden-card deal (C(47,2) * C(45,5) worlds before the first bet), which is
astronomically larger than a fixed budget of N sampled iterations.

Implementation choices that keep the constant factors low:
  * the game state is tiny (a few integers); only ONE state copy is made per iteration
    and moves are applied in place while descending the tree and during rollouts
  * legal actions and rollout probabilities come from precomputed lookup tables
  * the 7-card hand evaluator is a single O(7) counting pass (no 21-combination search)
  * the hidden-card sample uses a partial Fisher-Yates shuffle (O(cards needed), not O(52))

Usage:
    python poker_mcts.py                          # play a match vs. a random opponent
    python poker_mcts.py --opponent equity       # vs. a stronger equity-based baseline
    python poker_mcts.py --benchmark              # time-vs-iterations (linear scaling check)
    python poker_mcts.py --selftest               # sanity tests
"""

import argparse
import math
import random
import statistics
import time
from bisect import bisect_right

# --------------------------------------------------------------------------------------
# Game constants
# --------------------------------------------------------------------------------------
FOLD, CALL, RAISE = 0, 1, 2
ACTION_NAMES = {FOLD: "FOLD", CALL: "CALL/CHECK", RAISE: "RAISE/BET"}

ANTE = 1
BET_SIZE = (2, 2, 4, 4)          # fixed raise size on preflop, flop, turn, river
MAX_RAISES = 3                    # per street
MAX_CONTRIB = ANTE + MAX_RAISES * sum(BET_SIZE)
REWARD_SCALE = float(MAX_CONTRIB)  # payoffs are divided by this so rewards lie in [-1, 1]

RANKS = "23456789TJQKA"
SUITS = "cdhs"

# Legal actions indexed by  2 * (facing a bet) + (raises still allowed)
_LEGAL = (
    (CALL,),                 # no bet to call, raise cap reached
    (CALL, RAISE),           # no bet to call, can bet
    (FOLD, CALL),            # facing a bet, raise cap reached
    (FOLD, CALL, RAISE),     # facing a bet, can raise
)
# Cumulative rollout probabilities matching _LEGAL (rollouts rarely fold, often call)
_ROLLOUT_CUM = (
    (1.0,),
    (0.7, 1.0),
    (0.2, 1.0),
    (0.15, 0.75, 1.0),
)

rand = random.random


# --------------------------------------------------------------------------------------
# Cards and hand evaluation
# --------------------------------------------------------------------------------------
# A card is an int 0..51: rank = card >> 2 (0 = deuce ... 12 = ace), suit = card & 3.

def parse_cards(text):
    """'As Kd 9c' -> (51, 45, 29)"""
    return tuple(RANKS.index(t[0]) * 4 + SUITS.index(t[1]) for t in text.split())


def card_str(card):
    return RANKS[card >> 2] + SUITS[card & 3]


def _straight_low_bit(mask):
    """Return rank of the highest card of the best straight in a 13-bit rank mask, or -1."""
    ext = (mask << 1) | ((mask >> 12) & 1)       # bit 0 = ace playing low
    s = ext & (ext >> 1) & (ext >> 2) & (ext >> 3) & (ext >> 4)
    if not s:
        return -1
    return s.bit_length() + 2                     # lowest bit l -> high card rank l + 3


def _encode(category, ranks):
    score = category
    for i in range(5):
        score = score * 13 + (ranks[i] if i < len(ranks) else 0)
    return score


def evaluate(cards):
    """
    Score a 5-7 card hand in one O(7) pass. Higher score = better hand.
    Categories: 8 straight flush, 7 quads, 6 full house, 5 flush, 4 straight,
                3 trips, 2 two pair, 1 pair, 0 high card.
    """
    rank_count = [0] * 13
    suit_mask = [0, 0, 0, 0]
    suit_count = [0, 0, 0, 0]
    all_mask = 0
    for c in cards:
        r = c >> 2
        s = c & 3
        rank_count[r] += 1
        suit_mask[s] |= 1 << r
        suit_count[s] += 1
        all_mask |= 1 << r

    flush_mask = 0
    for s in range(4):
        if suit_count[s] >= 5:
            flush_mask = suit_mask[s]
            break

    if flush_mask:
        sf = _straight_low_bit(flush_mask)
        if sf >= 0:
            return _encode(8, (sf,))

    quads, trips, pairs, singles = [], [], [], []
    for r in range(12, -1, -1):
        n = rank_count[r]
        if n == 4:
            quads.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)
        elif n == 1:
            singles.append(r)

    if quads:
        kicker = max(trips + pairs + singles)
        return _encode(7, (quads[0], kicker))

    if trips and (len(trips) > 1 or pairs):
        best_pair = max(trips[1:] + pairs)
        return _encode(6, (trips[0], best_pair))

    if flush_mask:
        top = [r for r in range(12, -1, -1) if flush_mask >> r & 1][:5]
        return _encode(5, top)

    st = _straight_low_bit(all_mask)
    if st >= 0:
        return _encode(4, (st,))

    if trips:
        return _encode(3, [trips[0]] + singles[:2])

    if len(pairs) >= 2:
        kicker = max(pairs[2:] + singles)
        return _encode(2, (pairs[0], pairs[1], kicker))

    if pairs:
        return _encode(1, [pairs[0]] + singles[:3])

    return _encode(0, singles[:5])


# --------------------------------------------------------------------------------------
# Game state (betting only; cards live in a separate "world" tuple)
# --------------------------------------------------------------------------------------
class State:
    """Betting state of one hand. Small on purpose so copying/applying actions is cheap."""

    __slots__ = ("street", "to_act", "contrib", "raises", "acted", "done", "folder")

    def __init__(self):
        self.street = 0          # 0 preflop, 1 flop, 2 turn, 3 river
        self.to_act = 0
        self.contrib = [ANTE, ANTE]
        self.raises = 0          # raises made on this street
        self.acted = 0           # actions since the last raise (a raise counts as 1)
        self.done = False
        self.folder = -1

    def copy(self):
        s = State.__new__(State)
        s.street = self.street
        s.to_act = self.to_act
        s.contrib = self.contrib[:]
        s.raises = self.raises
        s.acted = self.acted
        s.done = self.done
        s.folder = self.folder
        return s

    def to_call(self):
        p = self.to_act
        return self.contrib[1 - p] - self.contrib[p]

    def index(self):
        """Index into the _LEGAL / _ROLLOUT_CUM tables."""
        p = self.to_act
        facing = 2 if self.contrib[1 - p] != self.contrib[p] else 0
        return facing + (1 if self.raises < MAX_RAISES else 0)

    def legal_actions(self):
        return _LEGAL[self.index()]

    def apply(self, action):
        """Apply an action in place."""
        p = self.to_act
        if action == FOLD:
            self.done = True
            self.folder = p
            return
        to_call = self.contrib[1 - p] - self.contrib[p]
        if action == CALL:
            self.contrib[p] += to_call
            self.acted += 1
            if self.acted >= 2:                      # betting round is over
                if self.street == 3:
                    self.done = True                 # showdown
                else:
                    self.street += 1
                    self.raises = 0
                    self.acted = 0
                    self.to_act = 0
                return
        else:                                        # RAISE
            self.contrib[p] += to_call + BET_SIZE[self.street]
            self.raises += 1
            self.acted = 1
        self.to_act = 1 - p

    def payoffs(self, world):
        """Chip payoff (player 0, player 1) of a finished hand. world = (hole0, hole1, board)."""
        c0, c1 = self.contrib
        if self.folder >= 0:
            return (-c0, c0) if self.folder == 0 else (c1, -c1)
        hole0, hole1, board = world
        s0 = evaluate(hole0 + board)
        s1 = evaluate(hole1 + board)
        if s0 > s1:
            return (c1, -c1)
        if s1 > s0:
            return (-c0, c0)
        return (0, 0)


# --------------------------------------------------------------------------------------
# Monte Carlo Tree Search
# --------------------------------------------------------------------------------------
class Node:
    __slots__ = ("children", "visits", "total", "player", "untried")

    def __init__(self, player):
        self.children = {}       # action -> Node
        self.visits = 0
        self.total = 0.0         # sum of normalized rewards for `player` (who moved into this node)
        self.player = player     # the player whose action led here (None at the root)
        self.untried = None      # actions not yet expanded (filled lazily)


class MCTSAgent:
    """
    Single-observer information-set MCTS with UCT selection.

    iterations  : number of MCTS iterations per decision (N in the complexity analysis)
    c           : UCT exploration constant (rewards are normalized to [-1, 1])
    """

    name = "MCTS"

    def __init__(self, iterations=1000, c=0.4):
        self.iterations = iterations
        self.c = c

    # ---- public API -------------------------------------------------------------
    def choose(self, state, player, hole, visible):
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]

        known = set(hole) | set(visible)
        unseen = [c for c in range(52) if c not in known]
        n_unseen = len(unseen)
        k = 2 + (5 - len(visible))               # opponent hole cards + missing board cards
        root = Node(None)

        for _ in range(self.iterations):
            # Determinization: partial Fisher-Yates shuffle, O(k) instead of O(52).
            for i in range(k):
                j = i + int(rand() * (n_unseen - i))
                unseen[i], unseen[j] = unseen[j], unseen[i]
            opp = (unseen[0], unseen[1])
            board = tuple(visible) + tuple(unseen[2:k])
            world = (hole, opp, board) if player == 0 else (opp, hole, board)
            self._iterate(root, state.copy(), world)

        # Robust child: most visited action
        return max(root.children.items(), key=lambda kv: kv[1].visits)[0]

    # ---- one MCTS iteration -----------------------------------------------------
    def _iterate(self, root, state, world):
        node = root
        path = [root]

        # 1) Selection + 2) Expansion
        while not state.done:
            if node.untried is None:
                node.untried = list(state.legal_actions())
            if node.untried:
                action = node.untried.pop()
                actor = state.to_act
                state.apply(action)
                child = Node(actor)
                node.children[action] = child
                path.append(child)
                break
            action, child = self._select(node, state.to_act)
            state.apply(action)
            path.append(child)
            node = child

        # 3) Simulation (random rollout to the end of the hand)
        while not state.done:
            i = state.index()
            state.apply(_LEGAL[i][bisect_right(_ROLLOUT_CUM[i], rand())])
        payoffs = state.payoffs(world)

        # 4) Backpropagation
        for n in path:
            n.visits += 1
            if n.player is not None:
                n.total += payoffs[n.player] / REWARD_SCALE

    def _select(self, node, actor):
        """UCT: argmax of mean reward + c * sqrt(ln(N_parent) / n_child). O(b)."""
        log_n = math.log(node.visits)
        c = self.c
        best_score = -1e18
        best = None
        for action, child in node.children.items():
            score = child.total / child.visits + c * math.sqrt(log_n / child.visits)
            if score > best_score:
                best_score = score
                best = (action, child)
        return best


# --------------------------------------------------------------------------------------
# Baseline opponents
# --------------------------------------------------------------------------------------
class RandomAgent:
    name = "Random"

    def choose(self, state, player, hole, visible):
        i = state.index()
        return _LEGAL[i][bisect_right(_ROLLOUT_CUM[i], rand())]


class CallingStation:
    name = "CallingStation"

    def choose(self, state, player, hole, visible):
        return CALL


class EquityAgent:
    """Estimates win probability by sampling, raises with strong hands, calls by pot odds."""

    name = "Equity"

    def __init__(self, samples=40, raise_threshold=0.7):
        self.samples = samples
        self.raise_threshold = raise_threshold

    def choose(self, state, player, hole, visible):
        known = set(hole) | set(visible)
        unseen = [c for c in range(52) if c not in known]
        k = 2 + (5 - len(visible))
        wins = 0.0
        for _ in range(self.samples):
            pick = random.sample(unseen, k)
            board = tuple(visible) + tuple(pick[2:])
            mine = evaluate(hole + board)
            theirs = evaluate(tuple(pick[:2]) + board)
            wins += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        equity = wins / self.samples

        if RAISE in state.legal_actions() and equity > self.raise_threshold:
            return RAISE
        to_call = state.to_call()
        if to_call == 0:
            return CALL
        pot = state.contrib[0] + state.contrib[1]
        return CALL if equity >= to_call / (pot + to_call) else FOLD


# --------------------------------------------------------------------------------------
# Playing hands and matches
# --------------------------------------------------------------------------------------
def play_hand(agents, verbose=False):
    """Play one hand. agents[0] sits in seat 0 (acts first), agents[1] in seat 1."""
    cards = random.sample(range(52), 9)
    holes = (tuple(cards[0:2]), tuple(cards[2:4]))
    board = tuple(cards[4:9])
    world = (holes[0], holes[1], board)
    state = State()
    if verbose:
        print("Hole cards:", [" ".join(map(card_str, h)) for h in holes],
              "| Board:", " ".join(map(card_str, board)))
    while not state.done:
        p = state.to_act
        vis = board[:(0, 3, 4, 5)[state.street]]
        action = agents[p].choose(state, p, holes[p], vis)
        if verbose:
            print(f"  street {state.street}: P{p} ({agents[p].name}) -> {ACTION_NAMES[action]}")
        state.apply(action)
    result = state.payoffs(world)
    if verbose:
        print("  payoffs:", result)
    return result


def play_match(agent, opponent, hands, verbose=False):
    """Play `hands` hands, swapping seats every hand so position luck cancels out."""
    results = []
    start = time.perf_counter()
    for h in range(hands):
        if h % 2 == 0:
            results.append(play_hand((agent, opponent), verbose and h < 3)[0])
        else:
            results.append(play_hand((opponent, agent), verbose and h < 3)[1])
    elapsed = time.perf_counter() - start
    mean = statistics.fmean(results)
    sem = statistics.stdev(results) / math.sqrt(len(results)) if len(results) > 1 else 0.0
    return mean, sem, elapsed


# --------------------------------------------------------------------------------------
# Time-complexity benchmark
# --------------------------------------------------------------------------------------
def benchmark(decisions=30, counts=(250, 500, 1000, 2000, 4000)):
    """Time one MCTS decision at several N. Linear growth confirms O(N) per decision."""
    print(f"{'iterations':>10} {'sec/decision':>14} {'us/iteration':>14} {'ratio vs prev':>14}")
    prev = None
    for n in counts:
        agent = MCTSAgent(iterations=n)
        total = 0.0
        for _ in range(decisions):
            cards = random.sample(range(52), 7)
            hole = tuple(cards[:2])
            visible = tuple(cards[2:5])               # flop decision
            state = State()
            state.street = 1
            start = time.perf_counter()
            agent.choose(state, 0, hole, visible)
            total += time.perf_counter() - start
        per_decision = total / decisions
        ratio = "" if prev is None else f"{per_decision / prev:.2f}x"
        print(f"{n:>10} {per_decision:>14.5f} {per_decision / n * 1e6:>14.2f} {ratio:>14}")
        prev = per_decision
    print("\nIterations double each row, so a ratio near 2.00x means time grows linearly: O(N).")


# --------------------------------------------------------------------------------------
# Self tests
# --------------------------------------------------------------------------------------
def selftest():
    ev = lambda s: evaluate(parse_cards(s))

    ordered = [
        ev("As Ks Qs Js Ts 2d 3c"),   # royal flush
        ev("9h 9d 9c 9s 2d 3c 4h"),   # quads
        ev("Kh Kd Kc 2s 2d 9c 4h"),   # full house
        ev("As 9s 7s 4s 2s Kd Qc"),   # flush
        ev("5h 6d 7c 8s 9d 2c Kh"),   # straight
        ev("Qh Qd Qc 2s 7d 9c 4h"),   # trips
        ev("Jh Jd 3c 3s 7d 9c 4h"),   # two pair
        ev("Ah Ad 3c 5s 7d 9c Kh"),   # pair
        ev("Ah Kd 3c 5s 7d 9c Jh"),   # high card
    ]
    assert ordered == sorted(ordered, reverse=True), "hand categories out of order"
    assert ev("As 2d 3c 4h 5s 9d Kc") < ev("2s 3d 4c 5h 6s 9d Kc"), "wheel must lose to 6-high straight"
    assert ev("As 2d 3c 4h 5s 9d Kc") > ev("Ah Ad 3c 5s 7d 9c Kh"), "wheel beats a pair"
    assert ev("Kh Kd Kc 2s 2d 2c 9h") > ev("Qh Qd Qc 2s 2d 9c 4h"), "KKK22 22 beats QQQ22"
    assert ev("Ah Ad 5c 6s 7d 9c Kh") > ev("Ah Ad 5c 6s 7d 9c Qh"), "kicker decides"
    assert ev("9s 8s 7s 6s 5s Ad Ac") > ev("Ah Ad Ac As 2d 3c 4h"), "straight flush beats quads"

    # Betting engine: every random hand terminates and payoffs are zero-sum
    for _ in range(2000):
        cards = random.sample(range(52), 9)
        world = (tuple(cards[0:2]), tuple(cards[2:4]), tuple(cards[4:9]))
        state = State()
        steps = 0
        while not state.done:
            state.apply(random.choice(state.legal_actions()))
            steps += 1
            assert steps <= 40, "hand did not terminate"
        p = state.payoffs(world)
        assert p[0] + p[1] == 0, "payoffs must be zero-sum"
        assert max(abs(p[0]), abs(p[1])) <= MAX_CONTRIB

    # MCTS must always return a legal action
    agent = MCTSAgent(iterations=200)
    for _ in range(50):
        cards = random.sample(range(52), 7)
        state = State()
        a = agent.choose(state, 0, tuple(cards[:2]), ())
        assert a in state.legal_actions()
    print("All self-tests passed.")


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MCTS Poker agent")
    parser.add_argument("--hands", type=int, default=200, help="hands to play in the match")
    parser.add_argument("--iterations", type=int, default=500, help="MCTS iterations per decision")
    parser.add_argument("--c", type=float, default=0.4, help="UCT exploration constant")
    parser.add_argument("--opponent", choices=["random", "station", "equity"], default="random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="print the first few hands")
    parser.add_argument("--benchmark", action="store_true", help="time vs. iterations")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.selftest:
        selftest()
        return
    if args.benchmark:
        benchmark()
        return

    opponent = {"random": RandomAgent, "station": CallingStation, "equity": EquityAgent}[args.opponent]()
    agent = MCTSAgent(iterations=args.iterations, c=args.c)
    mean, sem, elapsed = play_match(agent, opponent, args.hands, args.verbose)
    print(f"MCTS ({args.iterations} iterations) vs {opponent.name}: {args.hands} hands")
    print(f"  average result: {mean:+.3f} chips/hand  (+/- {sem:.3f} standard error)")
    print(f"  total time: {elapsed:.1f}s ({elapsed / args.hands * 1000:.1f} ms/hand)")


if __name__ == "__main__":
    main()
