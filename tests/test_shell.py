from giv.shell import parse_line


def test_state_with_choices():
    state, options = parse_line("I was charged $50 {billing, refund, other}")
    assert state == "I was charged $50"
    assert options == ["billing", "refund", "other"]


def test_choices_with_spaces_are_stripped():
    state, options = parse_line("text {  a , b ,c  }")
    assert state == "text"
    assert options == ["a", "b", "c"]


def test_no_braces_means_no_options():
    state, options = parse_line("just a plain state")
    assert state == "just a plain state"
    assert options is None


def test_state_after_braces_when_leading_empty():
    state, options = parse_line("{a, b} trailing state")
    assert state == "trailing state"
    assert options == ["a", "b"]


def test_empty_braces_yield_none():
    state, options = parse_line("text {}")
    assert state == "text"
    assert options is None
