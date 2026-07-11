"""Tests for eval_lib summary statistics.

Run with:
    pytest test_eval_lib.py -v
"""

from eval_lib import summarize_bucket


def test_avg_length_is_over_solved_episodes_only():
    # Two solved episodes (lengths 5 and 9) and one failure that ran to the
    # max_steps cutoff (100). avg_length must ignore the failure.
    bucket = {
        "returns": [1.4, 1.5, -0.6],
        "lengths": [5, 9, 100],
        "successes": [True, True, False],
    }
    s = summarize_bucket(bucket)
    assert s["avg_length"] == (5 + 9) / 2
    assert s["n"] == 3
    assert s["n_solved"] == 2


def test_avg_return_and_success_rate_are_over_all_episodes():
    bucket = {
        "returns": [1.0, 2.0, -1.0, 3.0],
        "lengths": [4, 6, 100, 8],
        "successes": [True, True, False, True],
    }
    s = summarize_bucket(bucket)
    assert s["avg_return"] == (1.0 + 2.0 - 1.0 + 3.0) / 4
    assert s["success_rate"] == 3 / 4


def test_avg_length_is_none_when_nothing_solved():
    bucket = {
        "returns": [-0.6, -0.7],
        "lengths": [100, 100],
        "successes": [False, False],
    }
    s = summarize_bucket(bucket)
    assert s["avg_length"] is None
    assert s["n_solved"] == 0
    assert s["success_rate"] == 0.0


def test_all_solved_matches_plain_mean_length():
    bucket = {
        "returns": [1.0, 1.0, 1.0],
        "lengths": [10, 12, 14],
        "successes": [True, True, True],
    }
    s = summarize_bucket(bucket)
    assert s["avg_length"] == (10 + 12 + 14) / 3
    assert s["n_solved"] == s["n"] == 3
