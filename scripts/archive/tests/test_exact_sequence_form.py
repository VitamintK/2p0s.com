from fractions import Fraction

from exact_sequence_form import nash_value


def test_kuhn_poker_nash_value_matches_known_exact_value():
    # The exact value of Kuhn poker to player 0 is a textbook result: -1/18.
    assert nash_value("kuhn_poker") == Fraction(-1, 18)
