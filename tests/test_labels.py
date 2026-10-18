from bharat_turn.labels import IPU, label_conversation, merge_ipus


def kinds(ev):
    return [(e.kind, e.speaker) for e in ev]


def test_merge_ipus_splits_on_200ms():
    ipus = merge_ipus([(0.0, 0.4, "main"), (0.5, 0.9, "kal"), (1.2, 1.6, "jaunga")], "A")
    assert [(u.start, u.end) for u in ipus] == [(0.0, 0.9), (1.2, 1.6)]
    assert ipus[0].text == "main kal"


def test_hold_then_shift():
    conv = [IPU("A", 0.0, 2.0, "haan dekhiye woh"), IPU("A", 2.65, 4.0, "matlab salary late"), IPU("B", 4.5, 6.0, "theek hai sir")]
    ev = label_conversation(conv)
    assert ("hold", "A") in kinds(ev)
    assert ("shift", "A") in kinds(ev)
    hold = next(e for e in ev if e.kind == "hold")
    assert (hold.start, hold.end) == (2.0, 2.65) and hold.prev_text.endswith("woh")


def test_backchannel_keeps_hold():
    conv = [IPU("A", 0.0, 3.0, "aapka emi due tha"), IPU("B", 3.2, 3.5, "haan"), IPU("A", 3.8, 6.0, "toh kab tak")]
    ev = label_conversation(conv)
    assert ("backchannel", "B") in kinds(ev)
    assert ("hold", "A") in kinds(ev)
    assert not any(e.kind == "shift" and e.start == 3.0 for e in ev)


def test_long_backchannel_is_not_backchannel():
    conv = [IPU("A", 0.0, 3.0, "x"), IPU("B", 3.2, 4.6, "haan haan haan haan"), IPU("A", 4.9, 6.0, "y")]
    ev = label_conversation(conv)
    assert ("backchannel", "B") not in kinds(ev)


def test_barge_in():
    conv = [IPU("A", 0.0, 5.0, "agent reading loan terms"), IPU("B", 4.0, 6.0, "nahi nahi ruko")]
    ev = label_conversation(conv)
    assert ("barge_in", "B") in kinds(ev)
