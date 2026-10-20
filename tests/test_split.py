"""eot-bench hi dev-test / sealed split (eval/split.py)."""

import pytest

from eval import split


def test_split_is_200_200_and_order_independent():
    a = split.load_assignment()
    assert len(a) == 400
    assert sum(h == "sealed" for h in a.values()) == 200
    ids = list(a)
    assert split.assign(ids) == a
    assert split.assign(list(reversed(ids))) == a


def test_guard_refuses_sealed_without_final():
    split.guard("devtest", final=False, what="test")  # allowed
    for half in ("sealed", "all"):
        with pytest.raises(SystemExit):
            split.guard(half, final=False, what="test")
