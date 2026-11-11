#!/usr/bin/env python3
"""
poker_multiplayer.py -- play fixed-limit Texas Hold'em against MCTS robots
===============================================================================
CSE 3500, Group 9: Fatima Tamanna, Jayson You, Luka Zorin, Kira Pyvovar

Needs poker_mcts.py in the SAME folder (it imports the hand evaluator, the
lookup tables, the belief sampler and the MCTS agent from there).
Pure Python 3, standard library only.

    python poker_multiplayer.py                    # you vs 3 robots, 10 hands
    python poker_multiplayer.py --players 6 --hands 20 --iterations 800
    python poker_multiplayer.py --benchmark        # robot decision timing, O(N)
    python poker_multiplayer.py --selftest

-------------------------------------------------------------------------------
GAME
-------------------------------------------------------------------------------
* 2..6 players (default 4: you plus 3 robots). Everyone antes 1 chip.
* Same fixed-limit structure as poker_mcts.py: streets preflop/flop/turn/
  river, bet sizes 2/2/4/4, at most 3 bets+raises per street.
* Actions: fold (only when facing a bet), check/call, bet/raise.
* The player who acts first ROTATES every hand (hand 1: seat 0, hand 2:
  seat 1, ...). On later streets the first player still in the hand at or
  after that seat acts first.
* A street ends when everyone still in has acted since the last raise and
  all bets are equal. If everyone but one folds, that player wins the pot.
* At showdown the best hand wins; ties SPLIT the pot. If the pot does not
  divide evenly, the leftover chips go one at a time to the tied winners
  nearest the first-to-act seat (the usual "odd chip" rule).
* Only net chips are tracked, so nobody can go broke: no blinds, no
  all-ins, no side pots.

-------------------------------------------------------------------------------
ROBOTS: multiplayer information-set MCTS
-------------------------------------------------------------------------------
Each robot runs the same MCTSAgent as the heads-up program (UCT selection,
expansion, rollout, backpropagation, most-visited final move). With N players:

* Determinization samples EVERY opponent's two hidden cards plus the unseen
  board each iteration. Each opponent's hand is drawn from a belief weighted
  by that opponent's betting so far (see BeliefSampler in poker_mcts.py),
  rejecting draws that clash with cards already handed out.
* Rewards are a vector with one entry per player: net chips divided by
  (players - 1) * 37, the most anyone can win, so every entry is in [-1, 1].
* Each node stores the reward of the player who moved INTO it (the max^n
  rule), so at every node UCT picks the move that is best for whoever is
  acting there.
* As in the heads-up agent, every opponent's decision node is split by that
  opponent's sampled hand-strength bucket, and the tree only grows through
  the current betting round (later rounds are rollouts). Both choices reduce
  "strategy fusion"; see the header of poker_mcts.py.

Complexity: one iteration is still O(b*d). The players add constant factors:
k = 2*(P-1) + 5 hidden cards to sample, P hand evaluations at showdown, and a
deeper (but bounded) betting tree. One decision is O(N) time and O(N) memory.
The belief setup costs O(P * H log H) once per decision, with H <= 1225.
"""

import argparse
import random
import statistics
import sys
import time

from poker_mcts import (
    ANTE, BET_SIZES, MAX_RAISES, MAX_CONTRIB, LEGAL, ROLLOUT_CDF, VISIBLE_BOARD,
    STREET_NAMES, FOLD, CALL, RAISE,
    evaluate, describe_hand, card_words, cards_from,
    partial_shuffle, BeliefSampler, MCTSAgent, HUState, new_deck, tree_size,
)


# =============================================================================
# N-player game state (same interface as HUState, so MCTSAgent works on it)
# =============================================================================
class NState:
    """Mutable N-player fixed-limit hand. Moves are applied in place.

    cards   : flat list, player p's hole cards at [2p, 2p+1], board at
              [2n .. 2n+4]. The whole board is dealt up front; only
              VISIBLE_BOARD[street] cards are face up.
    contrib : chips each player has put in (ante included)
    cur     : highest contribution so far (what everyone must match)
    pending : players who still have to act before the street can end
    first   : seat that acts first this hand (rotates every hand)
    """
    __slots__ = ("n", "cards", "contrib", "folded", "n_active", "street",
                 "to_act", "first", "cur", "raises", "pending", "terminal",
                 "_rewards", "history", "scale", "pbucket")

    @classmethod
    def new_hand(cls, n, deck, first=0):
        s = cls.__new__(cls)
        s.n = n
        s.cards = list(deck[:2 * n + 5])
        s.contrib = [ANTE] * n
        s.folded = [False] * n
        s.n_active = n
        s.street = 0
        s.first = first
        s.to_act = first
        s.cur = ANTE
        s.raises = 0
        s.pending = n
        s.terminal = False
        s._rewards = None
        s.history = []
        s.scale = float((n - 1) * MAX_CONTRIB)
        s.pbucket = None
        return s

    def copy(self):
        s = NState.__new__(NState)
        s.n = self.n
        s.cards = self.cards[:]
        s.contrib = self.contrib[:]
        s.folded = self.folded[:]
        s.n_active = self.n_active
        s.street = self.street
        s.first = self.first
        s.to_act = self.to_act
        s.cur = self.cur
        s.raises = self.raises
        s.pending = self.pending
        s.terminal = self.terminal
        s._rewards = None
        s.history = None
        s.scale = self.scale
        s.pbucket = self.pbucket
        return s

    @property
    def n_players(self):
        return self.n

    # ---- views ------------------------------------------------------------
    def hole(self, p):
        return self.cards[2 * p:2 * p + 2]

    def board(self):
        return self.cards[2 * self.n:2 * self.n + 5]

    def visible_board(self):
        b = 2 * self.n
        return self.cards[b:b + VISIBLE_BOARD[self.street]]

    def pot(self):
        return sum(self.contrib)

    def to_call(self):
        return self.cur - self.contrib[self.to_act]

    def legal_actions(self):
        return LEGAL[self.cur > self.contrib[self.to_act]][self.raises < MAX_RAISES]

    def rollout_action(self, u):
        for threshold, a in ROLLOUT_CDF[self.cur > self.contrib[self.to_act]][self.raises < MAX_RAISES]:
            if u < threshold:
                return a
        return CALL

    def _next_active(self, p):
        folded, n = self.folded, self.n
        p = (p + 1) % n
        while folded[p]:
            p = (p + 1) % n
        return p

    # ---- transitions -------------------------------------------------------
    def apply(self, a):
        """Apply action a for the player to act, in place. O(1) except the
        skip over folded seats, which is at most P steps."""
        p = self.to_act
        c = self.contrib
        if self.history is not None:
            self.history.append((self.street, p, int(self.cur > c[p]),
                                 int(self.raises < MAX_RAISES), a))
        if a == FOLD:
            self.folded[p] = True
            self.n_active -= 1
            self.pending -= 1
            if self.n_active == 1:
                self.terminal = True
                return
        elif a == CALL:
            c[p] = self.cur
            self.pending -= 1
        else:                                   # RAISE (or bet)
            self.cur += BET_SIZES[self.street]
            c[p] = self.cur
            self.raises += 1
            self.pending = self.n_active - 1    # everyone else must respond
        if self.pending == 0:                   # street over
            self.street += 1
            if self.street == 4:
                self.terminal = True
                return
            self.raises = 0
            self.pending = self.n_active
            self.to_act = self._next_active(self.first - 1)
        else:
            self.to_act = self._next_active(p)

    # ---- results -----------------------------------------------------------
    def showdown_order(self):
        """Players still in, in seat order starting from the first actor."""
        n = self.n
        return [(self.first + k) % n for k in range(n)
                if not self.folded[(self.first + k) % n]]

    def winners(self):
        """(list of winning seats, best score or None if won by folds)."""
        if self.n_active == 1:
            return [self.folded.index(False)], None
        board = self.board()
        best, win = -1, []
        for p in self.showdown_order():
            sc = evaluate(self.hole(p) + board)
            if sc > best:
                best, win = sc, [p]
            elif sc == best:
                win.append(p)
        return win, best

    def payoffs(self):
        """Net chips per player (integers, they always sum to zero)."""
        pot = sum(self.contrib)
        pay = [-x for x in self.contrib]
        win, _ = self.winners()
        share, rem = divmod(pot, len(win))
        for i, p in enumerate(win):              # odd chips: first-actor order
            pay[p] += share + (1 if i < rem else 0)
        return pay

    def rewards(self):
        r = self._rewards
        if r is None:
            sc = self.scale
            r = self._rewards = tuple(x / sc for x in self.payoffs())
        return r

    # ---- determinization ---------------------------------------------------
    def unseen_pool(self, observer):
        seen = set(self.hole(observer))
        seen.update(self.visible_board())
        return [c for c in range(52) if c not in seen]

    def make_sampler(self, observer):
        """Built once per decision: the unseen pool and one belief per
        opponent. Folded opponents cannot win, so their (irrelevant) cards
        are drawn uniformly; active ones use the betting-weighted belief."""
        pool = self.unseen_pool(observer)
        vis = self.visible_board()
        hist = self.history or []
        beliefs = []
        uniform = None
        for p in range(self.n):
            if p == observer:
                continue
            if not self.folded[p]:
                acts = [(st, f, cr, a) for st, pl, f, cr, a in hist if pl == p]
                beliefs.append((p, BeliefSampler(pool, vis, acts, self.street,
                                                 need_buckets=True)))
            else:
                if uniform is None:
                    uniform = BeliefSampler(pool, vis, [])
                beliefs.append((p, uniform))
        return pool, beliefs

    def determinize(self, observer, sampler, rnd):
        """Copy the state, then give every opponent two cards from their
        belief (redrawing on a clash with cards already given out) and fill
        the unseen board from what is left. One copy per iteration.
        O(P * log H + k) expected."""
        pool, beliefs = sampler
        s = self.copy()
        cards = s.cards
        used = set()
        pbucket = [0] * self.n
        for p, belief in beliefs:
            for _ in range(30):
                i = belief.sample_index(rnd())
                a, b = belief.hands[i]
                if a not in used and b not in used:
                    if belief.bucket_now is not None:
                        pbucket[p] = belief.bucket_now[i]
                    break
            else:                       # extremely rare: fall back to uniform
                partial_shuffle(pool, len(used) + 2, rnd)
                free = [c for c in pool if c not in used]
                a, b = free[0], free[1]
            used.add(a)
            used.add(b)
            cards[2 * p] = a
            cards[2 * p + 1] = b
        s.pbucket = pbucket
        vis = VISIBLE_BOARD[self.street]
        need = 5 - vis
        if need:
            partial_shuffle(pool, need + len(used), rnd)
            j = 2 * self.n + vis
            end = 2 * self.n + 5
            for c in pool:
                if c not in used:
                    cards[j] = c
                    j += 1
                    if j == end:
                        break
        return s


# =============================================================================
# Text helpers for the human player
# =============================================================================
COMING = (
    "5  (3 on the Flop, then 1 on the Turn, then 1 on the River)",
    "2  (1 on the Turn, then 1 on the River)",
    "1  (the River)",
    "none, all 5 table cards are out",
)
STREET_LONG = ("Preflop (betting round 1 of 4, no table cards yet)",
               "Flop (betting round 2 of 4)",
               "Turn (betting round 3 of 4)",
               "River (betting round 4 of 4, the last one)")

RULES = """
HOW THIS GAME WORKS (fixed-limit Texas Hold'em)
  * Everyone puts 1 chip (the "ante") into the pot at the start of each hand.
  * You get 2 private cards. Up to 5 shared "table cards" are dealt face up in
    three steps: the Flop (3 cards), the Turn (1 card), the River (1 card).
  * There is a round of betting before the Flop and after each step: 4 rounds.
  * Bets are a fixed size: 2 chips in the first two rounds, 4 chips in the
    last two. At most 3 bets/raises are allowed in a round.
  * Your hand is the best 5 cards you can make from your 2 cards plus the
    5 table cards. If more than one player is left after the River, everyone
    shows their cards and the best hand wins the whole pot (ties split it).
  * If everyone else folds, the last player left wins the pot right away.
  * Only your net chips are counted, so you can never go broke.

YOUR COMMANDS
  c = CHECK (stay in for free when nobody has bet), or
      CALL  (put in just enough chips to match the current bet)
  r = BET   (be the first to put chips in this round), or
      RAISE (match the current bet AND add one more bet on top)
  f = FOLD  (give up this hand; you lose what you already put in).
      Only offered when you face a bet; when checking is free, folding
      would just throw the hand away.
  h = HELP  (show these rules again)

HAND RANKING, best to worst
   1. Royal flush      A K Q J 10, all the same suit
   2. Straight flush   five in a row, all one suit (e.g. 5 6 7 8 9 of Hearts)
   3. Four of a kind   four cards of the same rank (e.g. four Queens)
   4. Full house       three of a kind plus a pair (e.g. three 8s, two Kings)
   5. Flush            any five cards of the same suit
   6. Straight         five ranks in a row, any suits (A 2 3 4 5 also counts)
   7. Three of a kind  three cards of the same rank
   8. Two pair         two different pairs
   9. Pair             two cards of the same rank
  10. High card        none of the above; the highest card plays

  "(all table cards: shared)" next to your hand means the table cards make
  that hand on their own, so every player still in has it too.
""".strip("\n")


def words(cards):
    return ", ".join(card_words(c) for c in cards) if cards else ""


def box(lines):
    """ASCII box (plain ASCII so it prints the same in every Windows console)."""
    width = max(len(x) for x in lines)
    edge = "+" + "-" * (width + 4) + "+"
    out = [edge] + ["|  " + x.ljust(width) + "  |" for x in lines] + [edge]
    return "\n".join(out)


def chips(n):
    return "%d chip%s" % (n, "" if n == 1 else "s")


def hand_text(hole, board=()):
    """'Pair  (Pair of Nines)' etc. for 2..7 cards. Adds a warning when the
    hand comes entirely from the shared table cards, because then every
    other player has it too (a classic beginner trap)."""
    sc = evaluate(list(hole) + list(board))
    text = describe_hand(sc)          # starts with the hand type: "Pair of ..."
    if len(board) >= 3 and sc >> 20 >= 1:
        b = evaluate(list(board))
        if b >> 20 == sc >> 20 and (b >> 16) & 15 == (sc >> 16) & 15:
            text += "  (all table cards: shared)"
    return text


def card_lines(label, cards, indent):
    """Print cards three per line so nothing runs past 80 columns."""
    for i in range(0, len(cards), 3):
        print((label if i == 0 else " " * len(label)) + words(cards[i:i + 3])
              + ("," if i + 3 < len(cards) else ""))


def describe_action(state, a, you=False):
    """What an action means, said BEFORE it is applied.
    you=True gives 'You call 2' instead of 'Robot 1 calls 2'."""
    owe = state.to_call()
    size = BET_SIZES[state.street]
    s = "" if you else "s"
    if a == FOLD:
        return "fold" + s
    if a == CALL:
        return ("check" + s) if owe == 0 else ("call%s %d" % (s, owe))
    if owe == 0:
        return "bet%s %d" % (s, size)
    return "raise%s (put%s in %d: %d to call + %d more)" % (s, s, owe + size, owe, size)


# =============================================================================
# The interactive game
# =============================================================================
class HumanSeat:
    name = "You"

    def __init__(self, game):
        self.game = game

    def act(self, state):
        return self.game.ask_human(state)


class Game:
    def __init__(self, n_players, hands, iterations, c, seed):
        self.n = n_players
        self.hands = hands
        self.rng = random.Random(seed)
        self.names = ["You"] + ["Robot %d" % i for i in range(1, n_players)]
        self.seats = [HumanSeat(self)] + [
            MCTSAgent(iterations, c, random.Random(seed * 101 + i), name=self.names[i])
            for i in range(1, n_players)]
        self.totals = [0] * n_players
        self.hand_no = 0

    # ---- the human's turn --------------------------------------------------
    def status_box(self, s):
        me = 0
        hole = s.hole(me)
        vis = s.visible_board()
        lines = [
            "YOUR TURN   (hand %d of %d)" % (self.hand_no, self.hands),
            "Street      : " + STREET_LONG[s.street],
            "Pot         : %d chips" % s.pot(),
            "To call     : " + chips(s.to_call()) +
            ("  (you can check for free)" if s.to_call() == 0 else ""),
            "Bets/raises : %d of %d used this round" % (s.raises, MAX_RAISES),
            "",
            "Your cards  : " + words(hole),
        ]
        if not vis:
            lines.append("Table cards : none yet (the Flop comes after this round)")
        for i in range(0, len(vis), 3):             # 3 cards per line max
            lines.append(("Table cards : " if i == 0 else "              ") +
                         words(vis[i:i + 3]) + ("," if i + 3 < len(vis) else ""))
        lines += [
            "More coming : " + COMING[s.street],
            "Your hand   : " + hand_text(hole, vis),
            "",
            "Players     :",
        ]
        for p in range(self.n):
            mark = "  <- first to act this hand" if p == s.first else ""
            status = "FOLDED" if s.folded[p] else "still in"
            lines.append("  %-9s put in %-9s %-8s%s"
                         % (self.names[p], chips(s.contrib[p]), status, mark))
        return box(lines)

    def ask_human(self, s):
        legal = s.legal_actions()
        owe = s.to_call()
        size = BET_SIZES[s.street]
        options = {}
        if FOLD in legal:
            options["f"] = (FOLD, "f = fold   (give up; you lose the %s you put in)"
                            % chips(s.contrib[0]))
        options["c"] = (CALL, "c = check  (put in nothing)" if owe == 0 else
                        "c = call   (put in %s to match)" % chips(owe))
        if RAISE in legal:
            options["r"] = (RAISE, "r = bet    (put in %d chips)" % size if owe == 0 else
                            "r = raise  (put in %d chips: %d to call + %d more)" % (owe + size, owe, size))
        print()
        print(self.status_box(s))
        while True:
            print("What do you want to do?")
            for key in ("f", "c", "r"):
                if key in options:
                    print("   " + options[key][1])
            print("   h = help (rules and hand rankings)")
            try:
                raw = input("Your choice (%s/h): " % "/".join(k for k in "fcr" if k in options))
            except EOFError:
                print("\n(no more input; checking/calling)")
                return CALL
            choice = raw.strip().lower()[:1]
            if choice == "h":
                print()
                print(RULES)
                print()
                print(self.status_box(s))
                continue
            if choice in options:
                return options[choice][0]
            if not choice:
                print("Please type a letter and press Enter: %s"
                      % ", ".join(list(options) + ["h"]))
            elif choice == "f":
                print("You can't fold right now: nobody has bet, so staying in is free. "
                      "Type c to check.")
            elif choice == "r":
                print("No more raises are allowed this round (the limit is %d)." % MAX_RAISES)
            else:
                print("Sorry, %r is not an option. Please type one of: %s"
                      % (raw.strip(), ", ".join(list(options) + ["h"])))

    # ---- one hand -----------------------------------------------------------
    def play_hand(self):
        self.hand_no += 1
        n = self.n
        first = (self.hand_no - 1) % n          # rotate who acts first
        s = NState.new_hand(n, new_deck(self.rng), first)
        print()
        print("=" * 70)
        print("HAND %d of %d.  Everyone antes 1 chip (pot: %d).  %s act%s first."
              % (self.hand_no, self.hands, s.pot(), self.names[first],
                 "" if first == 0 else "s"))
        print("Your cards: " + words(s.hole(0)))
        print("=" * 70)
        while not s.terminal:
            p = s.to_act
            street = s.street
            a = self.seats[p].act(s)
            if a not in s.legal_actions():
                raise RuntimeError("illegal action from " + self.names[p])
            text = describe_action(s, a, you=(p == 0))
            s.apply(a)
            print("  %-9s %s.   (pot is now %d)" % (self.names[p], text, s.pot()))
            if not s.terminal and s.street != street:
                vis = s.visible_board()
                new = vis[VISIBLE_BOARD[street]:]
                print()
                print("--- THE %s ---" % STREET_NAMES[s.street].upper())
                card_lines("    New table card%s: " % ("s " if len(new) > 1 else "  "), new, 0)
                card_lines("    All table cards : ", vis, 0)
                if not s.folded[0]:
                    print("    Your hand now   : " + hand_text(s.hole(0), vis))
        self.finish_hand(s)

    def finish_hand(self, s):
        n = self.n
        pay = s.payoffs()
        win, best = s.winners()
        print()
        if best is None:
            w = win[0]
            print("Everyone else folded. %s win%s the pot of %d chips."
                  % (self.names[w], "" if w == 0 else "s", s.pot()))
        else:
            print("SHOWDOWN!")
            card_lines("  Table cards: ", s.board(), 0)
            for p in s.showdown_order():
                print("  %-9s: %s" % (self.names[p], words(s.hole(p))))
                print("  %-9s  -> %s" % ("", describe_hand(evaluate(s.hole(p) + s.board()))))
            names = " and ".join(self.names[p] for p in win)
            if len(win) == 1:
                print("%s win%s the pot of %d chips with %s."
                      % (names, "" if win[0] == 0 else "s", s.pot(), describe_hand(best)))
            else:
                print("TIE! %s split the pot of %d chips (%s)." % (names, s.pot(),
                                                                    describe_hand(best)))
        for p in range(n):
            self.totals[p] += pay[p]
        print()
        print("This hand     : " + ",  ".join("%s %+d" % (self.names[p], pay[p]) for p in range(n)))
        print("Running total : " + ",  ".join("%s %+d" % (self.names[p], self.totals[p])
                                             for p in range(n)))

    def run(self):
        print(RULES)
        print()
        print("You are playing against %d MCTS robot%s. Good luck!"
              % (self.n - 1, "" if self.n == 2 else "s"))
        for h in range(self.hands):
            self.play_hand()
            if h < self.hands - 1:
                try:
                    ans = input("\nPress Enter for the next hand, or q then Enter to quit: ")
                except EOFError:
                    ans = "q"
                if ans.strip().lower().startswith("q"):
                    break
        print()
        print("=" * 70)
        print("FINAL TOTALS after %d hand%s (best to worst)"
              % (self.hand_no, "" if self.hand_no == 1 else "s"))
        for rank, p in enumerate(sorted(range(self.n), key=lambda q: -self.totals[q]), 1):
            print("  %d. %-9s %+d chips" % (rank, self.names[p], self.totals[p]))
        print("=" * 70)


# =============================================================================
# Benchmark: robot decision time vs N
# =============================================================================
def benchmark(n, c, seed, repeats=5):
    """Time one robot decision at increasing N for an n-player table."""
    from poker_mcts import preflop_equity
    preflop_equity(0, 4)                    # one-time lookup table
    rng = random.Random(seed)
    # Position: seat 0 bets preflop and everyone calls, so every opponent
    # has acted (and has a belief to build). Seat 0 then decides on the flop.
    s = NState.new_hand(n, new_deck(rng), 0)
    s.apply(RAISE)
    while s.street == 0:
        s.apply(CALL)
    print("Benchmark: one robot decision at a %d-player table (%s), median of %d runs"
          % (n, STREET_NAMES[s.street], repeats))
    print("'search' excludes the fixed per-decision setup (pool + %d beliefs)." % (n - 1))
    print("%10s %10s %10s %10s %8s %11s" % ("iterations", "setup ms", "search ms",
          "us/iter", "ratio", "tree nodes"))
    prev = None
    for it in (250, 500, 1000, 2000, 4000):
        times, setups = [], []
        for r in range(repeats):
            agent = MCTSAgent(it, c, random.Random(seed + r))
            t0 = time.perf_counter()
            root = agent.search(s, s.to_act)
            total = time.perf_counter() - t0
            setups.append(agent.last_setup_seconds)
            times.append(total - agent.last_setup_seconds)
        t = statistics.median(times)
        print("%10d %10.2f %10.1f %10.1f %8s %11d" % (
            it, 1000 * statistics.median(setups), 1000 * t, 1e6 * t / it,
            "-" if prev is None else "%.2fx" % (t / prev), tree_size(root)))
        prev = t
    print("Doubling N roughly doubles the search time: O(N) per decision, with the")
    print("number of players only changing the constant factor.")


# =============================================================================
# Self-test
# =============================================================================
class RandomMover:
    """Test helper: picks a uniformly random legal action (to generate
    thousands of varied hands for the engine tests)."""

    def __init__(self, rng):
        self.rng = rng
        self.name = "random-mover"

    def act(self, state):
        return self.rng.choice(state.legal_actions())


class Scripted:
    """Test helper: plays a fixed list of actions, then always checks/calls."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.name = "Scripted"

    def act(self, state):
        return self.actions.pop(0) if self.actions else CALL


def play_with(n, agents, deck, first=0):
    s = NState.new_hand(n, deck, first)
    steps = 0
    while not s.terminal:
        a = agents[s.to_act].act(s)
        assert a in s.legal_actions(), "illegal action"
        s.apply(a)
        steps += 1
        assert steps <= 4 * (n + MAX_RAISES * (n - 1)), "hand did not terminate"
    return s


def selftest(seed=777):
    rng = random.Random(seed)
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                               (" -- " + detail) if detail else ""))
        if not ok:
            failures += 1

    print("Random play, 2 to 6 players")
    for n in range(2, 7):
        bad = 0
        showdowns = folds = 0
        hands = 2000
        for h in range(hands):
            agents = [RandomMover(rng) for _ in range(n)]
            try:
                s = play_with(n, agents, new_deck(rng), h % n)
            except AssertionError:
                bad += 1
                continue
            pay = s.payoffs()
            live = [s.contrib[p] for p in range(n) if not s.folded[p]]
            if sum(pay) != 0 or (s.n_active > 1 and len(set(live)) != 1):
                bad += 1
            if s.n_active > 1:
                showdowns += 1
            else:
                folds += 1
        check("%d players: %d random hands terminate, zero-sum, equal live bets"
              % (n, hands), bad == 0, "%d showdowns, %d won by folds" % (showdowns, folds))

    print("Scripted situations")
    for n in (3, 4, 6):
        s = play_with(n, [Scripted([]) for _ in range(n)], new_deck(rng), 1)
        check("%d players all check down: everyone puts in exactly 1" % n,
              s.contrib == [1] * n and sum(s.payoffs()) == 0)
        agents = [Scripted([RAISE] * 4)] + [Scripted([]) for _ in range(n - 1)]
        s = play_with(n, agents, new_deck(rng), 0)
        check("%d players: seat 0 bets every street, everyone calls -> equal 13 each" % n,
              s.contrib == [13] * n, str(s.contrib))
        agents = [Scripted([RAISE])] + [Scripted([FOLD]) for _ in range(n - 1)]
        s = play_with(n, agents, new_deck(rng), 0)
        pay = s.payoffs()
        check("%d players: bet then fold-around -> bettor wins everyone's ante" % n,
              pay[0] == n - 1 and all(x == -1 for x in pay[1:]), str(pay))
        last = n - 1
        agents = [Scripted([CALL]) for _ in range(n - 1)] + [Scripted([RAISE])]
        for p in range(n - 1):
            agents[p] = Scripted([CALL, FOLD])
        s = play_with(n, agents, new_deck(rng), 0)
        pay = s.payoffs()
        check("%d players: checks to the LAST player, who bets; all fold -> +%d"
              % (n, n - 1), pay[last] == n - 1 and sum(pay) == 0, str(pay))

    # Split pot with an odd chip. Board is a royal flush, so everyone ties.
    board = cards_from("Ts Js Qs Ks As")
    holes = cards_from("2c 3d 4c 5d 6c 7d")
    deck = holes + board + [c for c in range(52) if c not in holes + board]
    agents = [Scripted([RAISE]), Scripted([CALL]), Scripted([FOLD])]
    s = play_with(3, agents, deck, 0)
    pay = s.payoffs()
    check("split pot: 7 chips between 2 tied players, odd chip to first seat",
          pay == [1, 0, -1], "payoffs %s, pot %d" % (pay, s.pot()))
    s = play_with(4, [Scripted([]) for _ in range(4)],
                  cards_from("2c 3d 4c 5d 6c 7d 8c 9d") + board + list(range(1)), 0)
    check("4-way tie on a royal-flush board: everyone gets their ante back",
          s.payoffs() == [0, 0, 0, 0], str(s.payoffs()))

    # First actor rotates and skips folded players on later streets.
    ok = all(NState.new_hand(5, new_deck(rng), f).to_act == f for f in range(5))
    s = NState.new_hand(4, new_deck(rng), 2)
    s.apply(RAISE)          # seat 2 bets
    s.apply(FOLD)           # seat 3 folds
    s.apply(CALL)           # seat 0 calls
    s.apply(CALL)           # seat 1 calls -> flop
    ok2 = s.street == 1 and s.to_act == 2
    s2 = NState.new_hand(4, new_deck(rng), 2)
    s2.apply(RAISE); s2.apply(CALL); s2.apply(CALL); s2.apply(CALL)
    ok3 = s2.street == 1 and s2.to_act == 2
    s3 = NState.new_hand(4, new_deck(rng), 3)
    s3.apply(RAISE); s3.apply(CALL); s3.apply(CALL); s3.apply(FOLD)   # seat 2 folds
    ok4 = s3.street == 1 and s3.to_act == 3
    check("first actor rotates and later streets start at the first live seat",
          ok and ok2 and ok3 and ok4)

    # The N-player engine with 2 players must agree with the heads-up engine.
    mismatch = 0
    for _ in range(1000):
        deck = new_deck(rng)
        a = HUState.new_hand(deck)
        b = NState.new_hand(2, deck, 0)
        while not a.terminal:
            if a.legal_actions() != b.legal_actions() or a.to_act != b.to_act:
                mismatch += 1
                break
            act = rng.choice(a.legal_actions())
            a.apply(act)
            b.apply(act)
        if not b.terminal or a.payoffs() != b.payoffs():
            mismatch += 1
    check("2-player engine matches poker_mcts.HUState on 1000 random hands",
          mismatch == 0, "%d mismatches" % mismatch)

    print("Robots")
    ok = True
    for n in (3, 6):
        s = NState.new_hand(n, new_deck(rng), 0)
        for _ in range(200):
            if s.terminal:
                s = NState.new_hand(n, new_deck(rng), rng.randrange(n))
            obs = s.to_act
            d = s.determinize(obs, s.make_sampler(obs), rng.random)
            if (d.hole(obs) != s.hole(obs) or d.visible_board() != s.visible_board()
                    or len(set(d.cards)) != len(d.cards)):
                ok = False
            s.apply(rng.choice(s.legal_actions()))
    check("determinization keeps the robot's view, no duplicate cards (3 and 6 players)", ok)
    illegal = tested = 0
    for i in range(150):
        n = 2 + i % 5
        s = NState.new_hand(n, new_deck(rng), rng.randrange(n))
        while not s.terminal and rng.random() < 0.85:
            s.apply(rng.choice(s.legal_actions()))
        if s.terminal:
            continue
        a = MCTSAgent(40, 0.5, random.Random(i)).act(s)
        tested += 1
        if a not in s.legal_actions():
            illegal += 1
    check("robots return a legal action in %d random positions (2-6 players)" % tested,
          illegal == 0)
    s = NState.new_hand(4, new_deck(rng), 0)
    root = MCTSAgent(800, 0.5, random.Random(5)).search(s, 0)
    r = s.copy()
    while not r.terminal:
        r.apply(RAISE if RAISE in r.legal_actions() else CALL)
    check("rewards are in [-1, 1] and visits equal N",
          root.visits == 800 and all(-1 <= x <= 1 for x in r.rewards()))

    print()
    print("SELFTEST %s (%d failure%s)" % ("PASSED" if failures == 0 else "FAILED",
                                          failures, "" if failures == 1 else "s"))
    return failures == 0


# =============================================================================
# Command line
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Play fixed-limit Texas Hold'em against MCTS robots (CSE 3500 Group 9)")
    ap.add_argument("--hands", type=int, default=10,
                    help="hands to play (default 10)")
    ap.add_argument("--iterations", type=int, default=500,
                    help="robot MCTS iterations per decision (default 500)")
    ap.add_argument("--players", type=int, default=4,
                    help="total players including you, 2-6 (default 4)")
    ap.add_argument("--c", type=float, default=1.0,
                    help="UCT exploration constant (default 1.0)")
    ap.add_argument("--benchmark", action="store_true",
                    help="time one robot decision at 250..4000 iterations")
    ap.add_argument("--selftest", action="store_true", help="run correctness tests")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    args = ap.parse_args(argv)
    if not 2 <= args.players <= 6:
        ap.error("--players must be between 2 and 6")
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)

    if args.selftest:
        return 0 if selftest() else 1
    if args.benchmark:
        benchmark(args.players, args.c, seed)
        return 0
    Game(args.players, args.hands, args.iterations, args.c, seed).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
