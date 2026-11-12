import io
import random
import unittest
from contextlib import redirect_stdout
from itertools import combinations
from unittest.mock import patch

import poker_multiplayer as mp
from poker_mcts import (CALL, FOLD, MAX_LOSS, RAISE, MCTSAgent, PokerState,
                        evaluate, hand_name, make_beliefs, new_deck,
                        parse_cards, sample_deal)


def score(text):
    return evaluate(parse_cards(text))


def slow_score5(cards):
    """Simple, obviously-correct 5-card scorer used to check evaluate()."""
    ranks = sorted((c // 4 for c in cards), reverse=True)
    counts = {r: ranks.count(r) for r in ranks}
    groups = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
    shape = sorted(counts.values(), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    straight = -1
    if len(counts) == 5:
        if ranks[0] - ranks[4] == 4:
            straight = ranks[0]
        elif ranks == [12, 3, 2, 1, 0]:
            straight = 3

    def pack(category, rs):
        return category << 20 | sum(r << (16 - 4 * i) for i, r in enumerate(rs))

    if flush and straight >= 0:
        return pack(8, [straight])
    if shape[0] == 4:
        return pack(7, groups)
    if shape == [3, 2]:
        return pack(6, groups)
    if flush:
        return pack(5, ranks)
    if straight >= 0:
        return pack(4, [straight])
    if shape[0] == 3:
        return pack(3, groups)
    if shape[:2] == [2, 2]:
        return pack(2, groups)
    if shape[0] == 2:
        return pack(1, groups)
    return pack(0, ranks)


class Scripted:
    """Plays a fixed list of actions, then always checks/calls."""

    def __init__(self, actions=()):
        self.actions = list(actions)

    def act(self, state):
        return self.actions.pop(0) if self.actions else CALL


class AlwaysRaise:
    def act(self, state):
        return RAISE if RAISE in state.legal_actions() else CALL


class RandomMover:
    def __init__(self, rng):
        self.rng = rng

    def act(self, state):
        return self.rng.choice(state.legal_actions())


def play(n, players, deck, first=0):
    s = PokerState(n, deck, first)
    steps = 0
    while not s.done:
        action = players[s.to_act].act(s)
        assert action in s.legal_actions()
        s.apply(action)
        steps += 1
        assert steps < 100, "hand did not end"
    return s


class TestHandEvaluator(unittest.TestCase):
    def test_hand_rank_ordering(self):
        hands = ["2c 5d 9h Js Kc 3d 7h",      # high card
                 "2c 2d 9h Js Kc 3d 7h",      # pair
                 "2c 2d 9h 9s Kc 3d 7h",      # two pair
                 "2c 2d 2h 9s Kc 3d 7h",      # three of a kind
                 "5c 6d 7h 8s 9c 2d Kh",      # straight
                 "2h 5h 9h Jh Kh 3d 7c",      # flush
                 "2c 2d 2h 9s 9c 3d 7h",      # full house
                 "2c 2d 2h 2s Kc 3d 7h",      # four of a kind
                 "5h 6h 7h 8h 9h 2d Kc"]      # straight flush
        scores = [score(h) for h in hands]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(len(set(scores)), 9)
        self.assertEqual(hand_name(parse_cards(hands[6])), "Full house")

    def test_wheel_straight(self):
        wheel = score("Ac 2d 3h 4s 5c Kd 9h")
        self.assertEqual(wheel >> 20, 4)
        self.assertLess(wheel, score("2d 3h 4s 5c 6h Kd 9h"))

    def test_kickers(self):
        self.assertGreater(score("Ac Ad Kh 9s 7c 4d 2h"), score("Ah As Qh 9s 7c 4d 2h"))
        self.assertEqual(score("Ac Kd 9h 7s 5c 3d 2h"), score("Ah Ks 9c 7d 5h 4c 2s"))

    def test_full_house_vs_full_house(self):
        self.assertGreater(score("Kc Kd Kh 2s 2c 5d 7h"), score("Qc Qd Qh As Ac 5d 7h"))
        self.assertGreater(score("Ac Ad Ah Kc Kd Ks 2h"), score("Ac Ad Ah Qc Qd Qs 2h"))

    def test_straight_flush_beats_quads(self):
        self.assertGreater(score("5h 6h 7h 8h 9h 9c 9d"), score("Ac Ad Ah As Kc 2d 3h"))

    def test_matches_brute_force(self):
        rng = random.Random(1)
        for _ in range(1000):
            seven = rng.sample(range(52), 7)
            best = max(slow_score5(five) for five in combinations(seven, 5))
            self.assertEqual(evaluate(seven), best)


class TestHeadsUp(unittest.TestCase):
    def test_random_hands_end_and_are_zero_sum(self):
        rng = random.Random(2)
        for _ in range(2000):
            s = play(2, [RandomMover(rng), RandomMover(rng)], new_deck(rng))
            self.assertEqual(sum(s.payoffs()), 0)
            self.assertLessEqual(max(s.bets), MAX_LOSS)

    def test_check_check_moves_to_flop(self):
        s = PokerState(2, new_deck(random.Random(3)))
        s.apply(CALL)
        self.assertEqual(s.street, 0)
        s.apply(CALL)
        self.assertEqual(s.street, 1)
        self.assertEqual(s.to_act, 0)

    def test_fold_loses_bets(self):
        s = PokerState(2, new_deck(random.Random(4)))
        s.apply(RAISE)
        s.apply(FOLD)
        self.assertTrue(s.done)
        self.assertEqual(s.payoffs(), [1, -1])

    def test_raise_cap(self):
        s = PokerState(2, new_deck(random.Random(5)))
        for _ in range(3):
            s.apply(RAISE)
        self.assertNotIn(RAISE, s.legal_actions())


class TestMultiplayer(unittest.TestCase):
    def test_random_hands_2_to_6_players(self):
        rng = random.Random(6)
        for n in range(2, 7):
            for h in range(1000):
                s = play(n, [RandomMover(rng) for _ in range(n)], new_deck(rng), h % n)
                self.assertEqual(sum(s.payoffs()), 0)
                live = {s.bets[p] for p in range(n) if not s.folded[p]}
                if s.folded.count(False) > 1:
                    self.assertEqual(len(live), 1)

    def test_everyone_calls_equal_contributions(self):
        for n in (3, 4, 6):
            s = play(n, [Scripted([RAISE] * 4)] + [Scripted() for _ in range(n - 1)],
                     new_deck(random.Random(n)))
            self.assertEqual(s.bets, [13] * n)

    def test_fold_around_wins_antes(self):
        for n in (3, 4, 6):
            # everyone checks to the last player, who bets, then everyone folds
            players = [Scripted([CALL, FOLD]) for _ in range(n - 1)] + [Scripted([RAISE])]
            s = play(n, players, new_deck(random.Random(n)))
            self.assertEqual(s.payoffs(), [-1] * (n - 1) + [n - 1])

    def test_split_pot_with_odd_chip(self):
        board = parse_cards("Ts Js Qs Ks As")             # royal flush: everyone ties
        holes = parse_cards("2c 3d 4c 5d 6c 7d")
        deck = holes + board
        s = play(3, [Scripted([RAISE]), Scripted(), Scripted([FOLD])], deck)
        self.assertEqual(s.pot(), 7)
        self.assertEqual(s.payoffs(), [1, 0, -1])         # 7 = 4 + 3, odd chip to seat 0

    def test_first_player_rotates(self):
        for first in range(4):
            self.assertEqual(PokerState(4, new_deck(random.Random(7)), first).to_act, first)
        s = PokerState(4, new_deck(random.Random(8)), first=3)
        s.apply(RAISE)      # seat 3
        s.apply(CALL)       # seat 0
        s.apply(CALL)       # seat 1
        s.apply(FOLD)       # seat 2
        self.assertEqual((s.street, s.to_act), (1, 3))


class TestMCTS(unittest.TestCase):
    def random_position(self, rng, n):
        s = PokerState(n, new_deck(rng), rng.randrange(n))
        while not s.done and rng.random() < 0.8:
            s.apply(rng.choice(s.legal_actions()))
        return None if s.done else s

    def test_always_legal(self):
        rng = random.Random(9)
        for i in range(60):
            s = self.random_position(rng, 2 + i % 5)
            if s:
                self.assertIn(MCTSAgent(50, 1.0, i).act(s), s.legal_actions())

    def test_visits_equal_iterations(self):
        s = PokerState(4, new_deck(random.Random(10)))
        root = MCTSAgent(300, 1.0, 1).search(s)
        self.assertEqual(root.visits, 300)
        self.assertEqual(sum(ch.visits for ch in root.children), 300)

    def test_rewards_in_range(self):
        rng = random.Random(11)
        for n in range(2, 7):
            s = play(n, [AlwaysRaise() for _ in range(n)], new_deck(rng))
            self.assertEqual(max(s.bets), MAX_LOSS)
            self.assertTrue(all(-1 <= r <= 1 for r in s.rewards()))

    def test_sampled_deal_keeps_what_player_sees(self):
        rng = random.Random(12)
        for n in (2, 6):
            for _ in range(50):
                s = self.random_position(rng, n)
                if not s:
                    continue
                me = s.to_act
                pool, beliefs = make_beliefs(s, me)
                d = sample_deal(s, me, pool, beliefs, rng)
                self.assertEqual(d.hole(me), s.hole(me))
                self.assertEqual(d.board(), s.board())
                self.assertEqual(len(set(d.cards)), len(d.cards))

    def test_does_not_fold_four_aces(self):
        s = PokerState(2, parse_cards("Ac Ad 2c 7d Ah As Kc Kd 3h"))
        s.apply(CALL)
        s.apply(CALL)       # flop: seat 0 has four Aces
        s.apply(CALL)
        s.apply(RAISE)      # opponent bets
        self.assertNotEqual(MCTSAgent(1000, 1.0, 2).act(s), FOLD)


class TestHumanInterface(unittest.TestCase):
    def make_game(self):
        game = mp.Game(4, 1, 30, 1.0, seed=5)
        game.hand_number = 1
        return game

    def test_status_box_spells_out_cards(self):
        game = self.make_game()
        s = PokerState(4, parse_cards("9c Ts 2d 3d 4h 5h 6c 7c Ah Kd 8s 2s 3s"))
        text = game.status(s)
        self.assertIn("Nine of Clubs, Ten of Spades", text)
        self.assertIn("none yet", text)
        self.assertIn("High card", text)

    def test_invalid_input_reprompts(self):
        game = self.make_game()
        s = PokerState(4, new_deck(random.Random(1)))
        with patch("builtins.input", side_effect=["x", "f", "h", "r"]), redirect_stdout(io.StringIO()):
            self.assertEqual(game.ask_human(s), RAISE)

    def test_full_game_runs(self):
        game = mp.Game(3, 2, 30, 1.0, seed=3)
        with patch("builtins.input", side_effect=["c"] * 50 + [""] * 5), redirect_stdout(io.StringIO()) as out:
            game.run()
        self.assertIn("FINAL TOTALS", out.getvalue())
        self.assertEqual(sum(game.totals), 0)


if __name__ == "__main__":
    unittest.main()