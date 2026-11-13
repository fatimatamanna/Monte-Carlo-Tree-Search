import argparse
import math
import random
import statistics
import time

from poker_mcts import CALL, FOLD, RAISE, MCTSAgent, PokerState, evaluate, new_deck

# Test opponents for the MCTS agent (heads-up only). Same seeds as in our report,
# so running this again gives the same numbers.


class RandomPlayer:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def act(self, state):
        return self.rng.choice(state.legal_actions())


class CallingStation:
    def act(self, state):
        return CALL                       # always checks or calls


class EquityPlayer:
    """Estimates its chance of winning from 200 random deals, raises when it is
    at least 65%, otherwise calls only when the pot odds are good enough."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def act(self, state):
        p = state.to_act
        mine, board = state.hole(p), state.board()
        pool = [c for c in range(52) if c not in mine + board]
        wins = 0.0
        for _ in range(200):
            cards = self.rng.sample(pool, 2 + 5 - len(board))
            full_board = board + cards[2:]
            a, b = evaluate(mine + full_board), evaluate(cards[:2] + full_board)
            wins += 1 if a > b else 0.5 if a == b else 0
        equity = wins / 200
        if equity >= 0.65 and RAISE in state.legal_actions():
            return RAISE
        owe = state.to_call()
        if owe == 0 or equity >= owe / (state.pot() + owe):
            return CALL
        return FOLD


def make_opponent(name):
    if name == "random":
        return RandomPlayer(2)
    if name == "station":
        return CallingStation()
    return EquityPlayer(2)


def play_match(opponent_name, iterations, c, hands):
    """Plays hands in pairs: the same cards twice with the seats swapped, which
    cancels out most of the luck. Returns (chips per hand, standard error, ms per hand)."""
    deal_rng = random.Random(2024)
    bot = MCTSAgent(iterations, c, 1)
    opponent = make_opponent(opponent_name)
    pair_results = []
    start = time.perf_counter()
    for _ in range(hands // 2):
        deck = new_deck(deal_rng)
        total = 0
        for seat in (0, 1):
            s = PokerState(2, deck)
            players = [bot, opponent] if seat == 0 else [opponent, bot]
            while not s.done:
                s.apply(players[s.to_act].act(s))
            total += s.payoffs()[seat]
        pair_results.append(total / 2)
    ms = 1000 * (time.perf_counter() - start) / (2 * len(pair_results))
    mean = statistics.fmean(pair_results)
    se = statistics.stdev(pair_results) / math.sqrt(len(pair_results))
    return mean, se, ms


def count_betting_sequences():
    """Counts every complete heads-up betting sequence by trying every action."""
    count = 0
    stack = [PokerState(2, list(range(9)))]
    while stack:
        s = stack.pop()
        for action in s.legal_actions():
            t = s.copy()
            t.apply(action)
            if t.done:
                count += 1
            else:
                stack.append(t)
    return count


def main():
    parser = argparse.ArgumentParser(description="Evaluate the MCTS agent against test opponents")
    parser.add_argument("--opponent", choices=["random", "station", "equity"], default="equity")
    parser.add_argument("--iterations", type=int, nargs="+", default=[500])
    parser.add_argument("--c", type=float, nargs="+", default=[1.0])
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--count", action="store_true", help="count betting sequences and deals")
    args = parser.parse_args()

    if args.count:
        print(f"Complete heads-up betting sequences: {count_betting_sequences():,}")
        print(f"Possible opponent hands, C(50, 2): {math.comb(50, 2):,}")
        print(f"Possible boards before the flop, C(48, 5): {math.comb(48, 5):,}")
        return

    print(f"MCTS vs {args.opponent}, {args.hands} hands per test")
    for c in args.c:
        for n in args.iterations:
            mean, se, ms = play_match(args.opponent, n, c, args.hands)
            verdict = "win" if mean > 2 * se else "loss" if mean < -2 * se else "within noise"
            print(f"N = {n:5}  C = {c:<4}  {mean:+.2f} chips/hand  SE {se:.2f}  "
                  f"{ms:.0f} ms/hand  ({verdict})")


if __name__ == "__main__":
    main()
