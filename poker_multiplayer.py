import argparse
import random
import textwrap
import time

from poker_mcts import (BET_SIZES, BOARD_SHOWN, CALL, FOLD, MAX_RAISES, RAISE,
                        STREETS, MCTSAgent, PokerState, card_name, hand_name,
                        new_deck)

RULES = """
HOW TO PLAY (fixed-limit Texas Hold'em)
  * Everyone puts 1 chip (the "ante") in the pot at the start of each hand.
  * You get 2 private cards. Then 5 shared table cards come out in three steps:
    the Flop (3 cards), the Turn (1 card) and the River (1 card).
  * There is a round of betting before the Flop and after each step.
  * Bets are 2 chips in the first two rounds and 4 chips in the last two.
    At most 3 bets/raises are allowed in one round.
  * Your hand is the best 5 cards out of your 2 cards plus the 5 table cards.
    If more than one player is left at the end, the best hand wins the pot
    (ties split it). If everyone else folds, you win the pot right away.

COMMANDS
  c = check (stay in for free) or call (match the current bet)
  r = bet (put chips in first) or raise (match the bet and add one more)
  f = fold (give up this hand; only offered when you face a bet)
  h = help (show this again)

HANDS, BEST TO WORST
  Straight flush   five in a row, all the same suit
  Four of a kind   four cards of the same rank
  Full house       three of a kind plus a pair
  Flush            five cards of the same suit
  Straight         five ranks in a row (A-2-3-4-5 counts)
  Three of a kind  three cards of the same rank
  Two pair         two different pairs
  Pair             two cards of the same rank
  High card        none of the above
"""

COMING = ["5 (3 on the Flop, 1 on the Turn, 1 on the River)",
          "2 (1 on the Turn, 1 on the River)",
          "1 (the River)",
          "none, all 5 are out"]


def names(cards):
    return ", ".join(card_name(c) for c in cards)


def chips(n):
    return f"{n} chip" + ("" if n == 1 else "s")


def box(lines):
    """Draw lines inside an ASCII box, wrapping anything too long."""
    wrapped = []
    for line in lines:
        wrapped += textwrap.wrap(line, 70, subsequent_indent=" " * 14) or [""]
    width = max(len(line) for line in wrapped)
    border = "+" + "-" * (width + 2) + "+"
    return "\n".join([border] + [f"| {line:<{width}} |" for line in wrapped] + [border])


def describe(state, action, you):
    """What an action means, said before it is applied ('raises 4', 'You call 2')."""
    owe = state.to_call()
    s = "" if you else "s"
    if action == FOLD:
        return "fold" + s
    if action == CALL:
        return f"call{s} {owe}" if owe else "check" + s
    if owe == 0:
        return f"bet{s} {BET_SIZES[state.street]}"
    return f"raise{s} (puts in {owe + BET_SIZES[state.street]})"


class Game:
    def __init__(self, players, hands, iterations, c, seed):
        self.n = players
        self.hands = hands
        self.rng = random.Random(seed)
        self.names = ["You"] + [f"Robot {i}" for i in range(1, players)]
        self.robots = [None] + [MCTSAgent(iterations, c, seed + i) for i in range(1, players)]
        self.totals = [0] * players
        self.hand_number = 0

    def status(self, s):
        board = s.board()
        lines = [
            f"YOUR TURN (hand {self.hand_number} of {self.hands})",
            f"Street      : {STREETS[s.street]} (betting round {s.street + 1} of 4)",
            f"Pot         : {chips(s.pot())}",
            f"To call     : {chips(s.to_call())}" + ("  (you can check for free)" if s.to_call() == 0 else ""),
            "",
            f"Your cards  : {names(s.hole(0))}",
            f"Table cards : {names(board) if board else 'none yet (the Flop comes after this round)'}",
            f"More coming : {COMING[s.street]}",
            f"Your hand   : {hand_name(s.hole(0) + board)}",
            "",
            "Players:",
        ]
        for p in range(self.n):
            state = "FOLDED" if s.folded[p] else "still in"
            first = "  (acts first)" if p == s.first else ""
            lines.append(f"  {self.names[p]:<8} put in {chips(s.bets[p]):<9} {state}{first}")
        return box(lines)

    def ask_human(self, s):
        owe = s.to_call()
        size = BET_SIZES[s.street]
        options = {}
        if FOLD in s.legal_actions():
            options["f"] = (FOLD, f"f = fold (you lose the {chips(s.bets[0])} you put in)")
        options["c"] = (CALL, f"c = call {chips(owe)}" if owe else "c = check (free)")
        if RAISE in s.legal_actions():
            options["r"] = (RAISE, f"r = raise (put in {chips(owe + size)})" if owe
                            else f"r = bet {chips(size)}")
        print("\n" + self.status(s))
        while True:
            for _, text in options.values():
                print("  " + text)
            print("  h = help")
            try:
                choice = input("Your choice: ").strip().lower()[:1]
            except EOFError:
                return CALL
            if choice in options:
                return options[choice][0]
            if choice == "h":
                print(RULES)
                print(self.status(s))
            elif choice == "f":
                print("You can't fold when checking is free. Type c to check.")
            elif choice == "r":
                print(f"No more raises allowed this round (the limit is {MAX_RAISES}).")
            else:
                print("Please type one of: " + ", ".join(list(options) + ["h"]))

    def play_hand(self):
        self.hand_number += 1
        first = (self.hand_number - 1) % self.n          # who acts first rotates
        s = PokerState(self.n, new_deck(self.rng), first)
        print("\n" + "=" * 60)
        print(f"HAND {self.hand_number} of {self.hands}. Everyone antes 1. "
              f"{self.names[first]} act{'' if first == 0 else 's'} first.")
        print(f"Your cards: {names(s.hole(0))}")
        while not s.done:
            p = s.to_act
            street = s.street
            if p == 0:
                action = self.ask_human(s)
            else:
                action = self.robots[p].act(s)
            text = describe(s, action, you=(p == 0))
            s.apply(action)
            print(f"  {self.names[p]} {text}.  (pot: {s.pot()})")
            if not s.done and s.street != street:
                new_cards = s.board()[BOARD_SHOWN[street]:]
                print(f"\n--- THE {STREETS[s.street].upper()} --- new: {names(new_cards)}")
                if not s.folded[0]:
                    print(f"    Your hand is now: {hand_name(s.hole(0) + s.board())}")
        self.show_result(s)

    def show_result(self, s):
        winners = s.winners()
        result = s.payoffs()
        if s.folded.count(False) == 1:
            print(f"\nEveryone else folded. {self.names[winners[0]]} wins {chips(s.pot())}.")
        else:
            board = s.cards[2 * self.n:]
            print("\nSHOWDOWN.")
            print(textwrap.fill("Table cards: " + names(board), 78, subsequent_indent="  "))
            for p in range(self.n):
                if not s.folded[p]:
                    print(f"  {self.names[p]:<8} {names(s.hole(p))} -> {hand_name(s.hole(p) + board)}")
            who = " and ".join(self.names[p] for p in winners)
            print(f"{who} {'split' if len(winners) > 1 else 'wins'} the pot of {chips(s.pot())}.")
        for p in range(self.n):
            self.totals[p] += result[p]
        for label, values in (("This hand:     ", result), ("Running total: ", self.totals)):
            text = label + ", ".join(f"{self.names[p]} {values[p]:+}" for p in range(self.n))
            print(textwrap.fill(text, 78, subsequent_indent=" " * 15))

    def run(self):
        print(RULES)
        for h in range(self.hands):
            self.play_hand()
            if h < self.hands - 1:
                try:
                    answer = input("\nPress Enter for the next hand (q to quit): ")
                except EOFError:
                    answer = "q"
                if answer.strip().lower() == "q":
                    break
        print("\nFINAL TOTALS")
        ranking = sorted(range(self.n), key=lambda p: -self.totals[p])
        for place, p in enumerate(ranking, 1):
            print(f"  {place}. {self.names[p]:<8} {self.totals[p]:+} chips")


def benchmark(players, c, seed):
    """Time one robot decision on the flop after everyone called a bet."""
    s = PokerState(players, new_deck(random.Random(seed)))
    s.apply(RAISE)
    while s.street == 0:
        s.apply(CALL)
    print(f"Time for one robot decision with {players} players (best of 3)")
    print(f"{'iterations':>10} {'ms':>8} {'ratio':>7}")
    previous = None
    for n in (250, 500, 1000, 2000, 4000):
        times = []
        for run in range(3):
            start = time.perf_counter()
            MCTSAgent(n, c, seed + run).search(s)
            times.append(time.perf_counter() - start)
        t = min(times)
        ratio = f"{t / previous:.2f}x" if previous else "-"
        print(f"{n:>10} {1000 * t:>8.1f} {ratio:>7}")
        previous = t


def main():
    parser = argparse.ArgumentParser(description="Play limit Texas Hold'em against MCTS robots")
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--players", type=int, default=4, choices=range(2, 7))
    parser.add_argument("--c", type=float, default=1.0, help="UCT exploration constant")
    parser.add_argument("--seed", type=int, default=random.randrange(10**6))
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        benchmark(args.players, args.c, args.seed)
    else:
        Game(args.players, args.hands, args.iterations, args.c, args.seed).run()


if __name__ == "__main__":
    main()