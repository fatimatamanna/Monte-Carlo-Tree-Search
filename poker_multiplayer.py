"""
Human vs 3 MCTS robots - 4-player limit Texas Hold'em
CSE 3500 - Group 9

Builds on poker_mcts.py (keep both files in the same folder). That file has the heads-up
version; this one generalizes the game engine and the MCTS agent to N players (default 4).

Rules (same fixed-limit style as poker_mcts.py):
  * Everyone antes 1 chip, gets 2 hole cards; 5 community cards over flop / turn / river.
  * Actions: FOLD, CALL (call or check), RAISE (bet or raise by the street's fixed size:
    2 preflop, 2 flop, 4 turn, 4 river). At most 3 raises per street.
  * The player who acts first rotates every hand. Pot goes to the best hand at showdown
    (split on ties) or to the last player left if everyone else folds.
  * Chips are tracked as a running net win/loss, so nobody ever busts.

Robots use single-observer information-set MCTS with UCT. Each iteration samples every
opponent's hidden cards plus the unseen board, then runs Selection, Expansion, Simulation
and Backpropagation. Each node stores the reward of the player who moved into it, so
selection maximizes the reward of whoever is acting (the max^n rule for multiplayer games).

Time complexity: one iteration is O(b * d) with b <= 3 actions and d <= O(P * 4 * (R + 1))
betting steps (P players, R max raises), a constant for a fixed table. One robot decision
costs O(I * b * d) = O(I) for I iterations, linear in I, with O(I) memory. The only extra
cost versus heads-up is the constant factor P.

Usage:
    python poker_multiplayer.py                   # you vs 3 robots (type f / c / r)
    python poker_multiplayer.py --iterations 1000 # stronger (slower) robots
    python poker_multiplayer.py --benchmark       # robots vs a random player, no human
    python poker_multiplayer.py --selftest
"""

import argparse
import math
import random
import sys
import time
from bisect import bisect_right

from poker_mcts import (
    FOLD, CALL, RAISE, ANTE, BET_SIZE, MAX_RAISES, MAX_CONTRIB,
    evaluate, card_str, rand, Node, _LEGAL, _ROLLOUT_CUM,
)

HAND_NAMES = ("High card", "Pair", "Two pair", "Three of a kind", "Straight",
              "Flush", "Full house", "Four of a kind", "Straight flush")
STREET_NAMES = ("Preflop", "Flop", "Turn", "River")
BOARD_SIZE = (0, 3, 4, 5)         # community cards visible on each street


def hand_name(cards):
    return HAND_NAMES[evaluate(cards) // 13 ** 5]


RANK_WORDS = ("Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
              "Jack", "Queen", "King", "Ace")
SUIT_WORDS = ("Clubs", "Diamonds", "Hearts", "Spades")      # same order as SUITS in poker_mcts


def card_long(card):
    """51 -> 'Ace of Spades'"""
    return f"{RANK_WORDS[card >> 2]} of {SUIT_WORDS[card & 3]}"


def cards_text(cards):
    return ", ".join(card_long(c) for c in cards)


HELP_TEXT = """
HOW TO PLAY
  Everyone gets 2 private cards. Up to 5 shared cards (the "table cards") are dealt in
  stages: the Flop (3 cards), the Turn (1 card) and the River (1 card).
  Your hand is the best 5 cards you can make from your 2 cards plus the table cards.
  When betting ends, the best hand left wins the pot. If everyone else folds, you win it.

  f = fold    give up this hand (you lose what you already put in)
  c = check   stay in for free when nobody has bet, or call to match a bet
  r = bet     put in more chips (bet when nobody has bet, raise when someone has)
  h = help    show this message again

HAND RANKS (best to worst)
  Straight flush > Four of a kind > Full house > Flush > Straight
  > Three of a kind > Two pair > Pair > High card
"""


# --------------------------------------------------------------------------------------
# N-player betting state
# --------------------------------------------------------------------------------------
class MultiState:
    """Betting state for N players. Cards live in a separate world = (holes, board)."""

    __slots__ = ("n", "first", "street", "to_act", "contrib", "folded",
                 "raises", "pending", "active", "done")

    def __init__(self, n, first=0):
        self.n = n
        self.first = first               # seat that opens the betting each street
        self.street = 0
        self.to_act = first
        self.contrib = [ANTE] * n
        self.folded = [False] * n
        self.raises = 0                  # raises made on this street
        self.pending = n                 # players who still have to act this street
        self.active = n                  # players who have not folded
        self.done = False

    def copy(self):
        s = MultiState.__new__(MultiState)
        s.n = self.n
        s.first = self.first
        s.street = self.street
        s.to_act = self.to_act
        s.contrib = self.contrib[:]
        s.folded = self.folded[:]
        s.raises = self.raises
        s.pending = self.pending
        s.active = self.active
        s.done = self.done
        return s

    def to_call(self):
        return max(self.contrib) - self.contrib[self.to_act]

    def index(self):
        """Index into the _LEGAL / _ROLLOUT_CUM tables from poker_mcts."""
        facing = 2 if self.contrib[self.to_act] < max(self.contrib) else 0
        return facing + (1 if self.raises < MAX_RAISES else 0)

    def legal_actions(self):
        return _LEGAL[self.index()]

    def _next_active(self, p):
        folded = self.folded
        n = self.n
        for i in range(1, n + 1):
            q = (p + i) % n
            if not folded[q]:
                return q
        return p

    def apply(self, action):
        """Apply an action in place."""
        p = self.to_act
        if action == FOLD:
            self.folded[p] = True
            self.active -= 1
            if self.active == 1:
                self.done = True
                return
            self.pending -= 1
        else:
            top = max(self.contrib)
            if action == CALL:
                self.contrib[p] = top
                self.pending -= 1
            else:                                    # RAISE
                self.contrib[p] = top + BET_SIZE[self.street]
                self.raises += 1
                self.pending = self.active - 1       # everyone else must respond

        if self.pending == 0:                        # betting round finished
            if self.street == 3:
                self.done = True                     # showdown
                return
            self.street += 1
            self.raises = 0
            self.pending = self.active
            first = self.first
            self.to_act = first if not self.folded[first] else self._next_active(first)
            return
        self.to_act = self._next_active(p)

    def payoffs(self, world):
        """Net chips won/lost by each player. Zero-sum. world = (holes, board)."""
        contrib = self.contrib
        pot = sum(contrib)
        result = [-c for c in contrib]
        if self.active == 1:
            winner = self.folded.index(False)
            result[winner] += pot
            return result
        holes, board = world
        best = -1
        winners = []
        for i in range(self.n):
            if self.folded[i]:
                continue
            score = evaluate(holes[i] + board)
            if score > best:
                best = score
                winners = [i]
            elif score == best:
                winners.append(i)
        share = pot / len(winners)
        for i in winners:
            result[i] += share
        return result


# --------------------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------------------
class MultiMCTSAgent:
    """Information-set MCTS (UCT) for N players."""

    name = "MCTS"

    def __init__(self, iterations=500, c=0.4):
        self.iterations = iterations
        self.c = c

    def choose(self, state, player, hole, visible):
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]

        n = state.n
        known = set(hole) | set(visible)
        unseen = [c for c in range(52) if c not in known]
        n_unseen = len(unseen)
        opp_cards = 2 * (n - 1)
        k = opp_cards + (5 - len(visible))
        scale = float(MAX_CONTRIB * (n - 1))         # keeps rewards within [-1, 1]
        root = Node(None)

        for _ in range(self.iterations):
            # Determinization via partial Fisher-Yates shuffle: O(cards needed).
            for i in range(k):
                j = i + int(rand() * (n_unseen - i))
                unseen[i], unseen[j] = unseen[j], unseen[i]
            holes = [None] * n
            idx = 0
            for q in range(n):
                if q == player:
                    holes[q] = hole
                else:
                    holes[q] = (unseen[idx], unseen[idx + 1])
                    idx += 2
            board = tuple(visible) + tuple(unseen[opp_cards:k])
            self._iterate(root, state.copy(), (holes, board), scale)

        return max(root.children.items(), key=lambda kv: kv[1].visits)[0]

    def _iterate(self, root, state, world, scale):
        node = root
        path = [root]

        # Selection + Expansion
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
            action, child = self._select(node)
            state.apply(action)
            path.append(child)
            node = child

        # Simulation
        while not state.done:
            i = state.index()
            state.apply(_LEGAL[i][bisect_right(_ROLLOUT_CUM[i], rand())])
        payoffs = state.payoffs(world)

        # Backpropagation
        for nd in path:
            nd.visits += 1
            if nd.player is not None:
                nd.total += payoffs[nd.player] / scale

    def _select(self, node):
        """UCT over the children of `node`. O(b)."""
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


class RandomAgent:
    name = "Random"

    def choose(self, state, player, hole, visible):
        i = state.index()
        return _LEGAL[i][bisect_right(_ROLLOUT_CUM[i], rand())]


class HumanAgent:
    """Console player: type f (fold), c (call/check) or r (bet/raise)."""

    name = "You"

    def __init__(self, names):
        self.names = names

    def _show(self, state, hole, visible):
        """Print everything the human needs to decide: own cards, table cards, pot, players."""
        pot = sum(state.contrib)
        print("\n" + "=" * 60)
        print(f" YOUR TURN | {STREET_NAMES[state.street]} | Pot: {pot} | To call: {state.to_call()}")
        print("=" * 60)
        print(f" Your cards:     {cards_text(hole)}")
        if visible:
            more = 5 - len(visible)
            note = f"   ({more} more shared card{'s' if more != 1 else ''} to come)" if more else ""
            print(f" Table cards:    {cards_text(visible)}{note}")
        else:
            print(" Table cards:    none yet (3 shared cards, the Flop, come after this round)")
        print(f" Your hand now:  {hand_name(tuple(hole) + tuple(visible))}")
        print(" Players:")
        for q in range(state.n):
            status = "FOLDED" if state.folded[q] else "in hand"
            print(f"   {self.names[q]:<7} has put in {state.contrib[q]:>3}   {status}")

    def choose(self, state, player, hole, visible):
        legal = state.legal_actions()
        to_call = state.to_call()
        self._show(state, hole, visible)

        options = []
        if FOLD in legal:
            options.append("[f] fold")
        options.append("[c] call " + str(to_call) if to_call else "[c] check")
        if RAISE in legal:
            amount = to_call + BET_SIZE[state.street]
            options.append(f"[r] {'raise' if to_call else 'bet'} (puts in {amount})")
        options.append("[h] help")
        prompt = "Your move: " + "   ".join(options) + "\n> "

        while True:
            try:
                text = input(prompt).strip().lower()
            except EOFError:
                print("\nInput closed, exiting.")
                sys.exit(0)
            if text[:1] in ("h", "?"):
                print(HELP_TEXT)
                self._show(state, hole, visible)
                continue
            if text[:1] == "f" and FOLD in legal:
                return FOLD
            if text[:1] in ("c", "k"):
                return CALL
            if text[:1] in ("r", "b") and RAISE in legal:
                return RAISE
            if text[:1] == "f":
                print("Nothing to call, so you can just check ('c').")
            elif text[:1] in ("r", "b"):
                print("Raise cap reached this street. Choose call or fold.")
            else:
                print("Please type f, c or r.")


# --------------------------------------------------------------------------------------
# Running a table
# --------------------------------------------------------------------------------------
def describe(name, action, to_call, street):
    s = "" if name == "You" else "s"
    if action == FOLD:
        return f"{name} fold{s}"
    if action == CALL:
        return f"{name} check{s}" if to_call == 0 else f"{name} call{s} {to_call}"
    amount = to_call + BET_SIZE[street]
    return f"{name} bet{s} {amount}" if to_call == 0 else f"{name} raise{'' if s == '' else 's'} (puts in {amount})"


def play_table(agents, names, hands, verbose=True, interactive=False):
    """Play `hands` hands. Returns the net chips won by each seat."""
    n = len(agents)
    net = [0.0] * n
    for h in range(hands):
        cards = random.sample(range(52), 2 * n + 5)
        holes = tuple(tuple(cards[2 * i:2 * i + 2]) for i in range(n))
        board = tuple(cards[2 * n:2 * n + 5])
        world = (holes, board)
        state = MultiState(n, first=h % n)

        if verbose:
            print(f"\n{'=' * 56}\nHand {h + 1} of {hands}   (first to act: {names[state.first]})")

        while not state.done:
            p = state.to_act
            visible = board[:BOARD_SIZE[state.street]]
            to_call = state.to_call()
            street_before = state.street
            action = agents[p].choose(state, p, holes[p], visible)
            if verbose:
                print("  >> " + describe(names[p], action, to_call, state.street))
            state.apply(action)
            if verbose and not state.done and state.street != street_before:
                print(f"\n*** {STREET_NAMES[state.street]} dealt. Table cards: " +
                      cards_text(board[:BOARD_SIZE[state.street]]) + " ***")

        result = state.payoffs(world)
        for i in range(n):
            net[i] += result[i]

        if verbose:
            pot = sum(state.contrib)
            if state.active == 1:
                winner = state.folded.index(False)
                print(f"\n{names[winner]} won the pot of {pot:g} (everyone else folded).")
            else:
                print("\nSHOWDOWN. Table cards: " + cards_text(board))
                for i in range(n):
                    if not state.folded[i]:
                        print(f"  {names[i]:<7} {cards_text(holes[i])}  ->  "
                              f"{hand_name(holes[i] + board)}")
            print("Hand result:  " + "   ".join(f"{names[i]} {result[i]:+g}" for i in range(n)))
            print("Running total: " + "   ".join(f"{names[i]} {net[i]:+g}" for i in range(n)))

        if interactive and h < hands - 1:
            try:
                if input("\nPress Enter for the next hand (q to quit): ").strip().lower() == "q":
                    break
            except EOFError:
                break
    return net


# --------------------------------------------------------------------------------------
# Self tests and benchmark
# --------------------------------------------------------------------------------------
def selftest():
    for n in range(2, 7):
        for _ in range(1500):
            cards = random.sample(range(52), 2 * n + 5)
            holes = tuple(tuple(cards[2 * i:2 * i + 2]) for i in range(n))
            world = (holes, tuple(cards[2 * n:]))
            state = MultiState(n, first=random.randrange(n))
            steps = 0
            while not state.done:
                state.apply(random.choice(state.legal_actions()))
                steps += 1
                assert steps <= 300, "hand did not terminate"
            pay = state.payoffs(world)
            assert abs(sum(pay)) < 1e-9, "payoffs must be zero-sum"
            assert max(abs(x) for x in pay) <= MAX_CONTRIB * (n - 1) + 1e-9
    # Everybody calls down: all contribute equally, no one can lose more than the ante
    state = MultiState(4, first=1)
    while not state.done:
        state.apply(CALL)
    assert state.contrib == [1, 1, 1, 1] and state.street == 3
    # Fold around: last player standing wins everyone else's ante
    state = MultiState(4, first=0)
    for _ in range(3):
        state.apply(FOLD)
    assert state.done and state.payoffs(None) == [-1, -1, -1, 3]
    # Robots always return a legal action
    agent = MultiMCTSAgent(iterations=150)
    for n in (3, 4):
        for _ in range(30):
            cards = random.sample(range(52), 7)
            state = MultiState(n, first=random.randrange(n))
            p = state.to_act
            assert agent.choose(state, p, tuple(cards[:2]), ()) in state.legal_actions()
    print("All multiplayer self-tests passed.")


def benchmark(hands, iterations, players):
    """Seat 0 plays randomly, everyone else is an MCTS robot. Robots should win chips."""
    names = ["Random"] + [f"Bot-{i}" for i in range(1, players)]
    agents = [RandomAgent()] + [MultiMCTSAgent(iterations) for _ in range(players - 1)]
    start = time.perf_counter()
    net = play_table(agents, names, hands, verbose=False)
    elapsed = time.perf_counter() - start
    print(f"{players} players, {hands} hands, {iterations} iterations per robot decision")
    for name, chips in zip(names, net):
        print(f"  {name:<7} {chips:+9.1f} chips   ({chips / hands:+.2f} per hand)")
    print(f"  time: {elapsed:.1f}s ({elapsed / hands * 1000:.1f} ms/hand)")


def main():
    parser = argparse.ArgumentParser(description="Human vs MCTS robots poker")
    parser.add_argument("--hands", type=int, default=None, help="number of hands to play")
    parser.add_argument("--iterations", type=int, default=500, help="MCTS iterations per robot decision")
    parser.add_argument("--players", type=int, default=4, help="players at the table (2-6)")
    parser.add_argument("--benchmark", action="store_true", help="robots vs a random player, no human")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    if not 2 <= args.players <= 6:
        parser.error("--players must be between 2 and 6")

    if args.selftest:
        selftest()
        return
    if args.benchmark:
        benchmark(args.hands or 200, args.iterations, args.players)
        return

    hands = args.hands or 10
    names = ["You"] + [f"Bot-{i}" for i in range(1, args.players)]
    agents = [HumanAgent(names)] + [MultiMCTSAgent(args.iterations) for _ in range(args.players - 1)]
    print(f"Texas Hold'em: you vs {args.players - 1} MCTS robots, {hands} hands. Ante {ANTE}, "
          f"bet sizes {BET_SIZE}. Everyone starts even; totals are net chips.")
    print(HELP_TEXT)
    net = play_table(agents, names, hands, verbose=True, interactive=True)
    print("\n" + "=" * 56 + "\nFinal totals:")
    for name, chips in sorted(zip(names, net), key=lambda x: -x[1]):
        print(f"  {name:<7} {chips:+g}")


if __name__ == "__main__":
    main()