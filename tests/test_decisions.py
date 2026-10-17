import pytest

from giv import Decision, Question
from giv.decisions import QuestionError
from giv.results import Answer


def test_choice_basic():
    q = Question.choice("Which team?", ["billing", "technical", "other"])
    assert q.kind == "choice"
    assert q.options == ("billing", "technical", "other")
    assert q.name == "which_team"


def test_choice_needs_two_options():
    with pytest.raises(QuestionError):
        Question.choice("Which?", ["only_one"])


def test_choice_rejects_duplicates():
    with pytest.raises(QuestionError):
        Question.choice("Which?", ["a", "a"])


def test_noul_defaults():
    q = Question.noul("Is this urgent?", name="urgent")
    assert q.kind == "noul"
    assert q.options == ("Yes", "No")
    assert q.name == "urgent"


def test_score_bins():
    q = Question.score("How urgent?", bins=5, name="urgency")
    assert q.kind == "score"
    assert len(q.options) == 5
    assert q.is_ordered


def test_score_levels_centers():
    q = Question.score("Risk?", levels=["low", "medium", "high"], scale=(0, 1), name="risk")
    assert q.options == ("low", "medium", "high")
    assert q.centers == (0.0, 0.5, 1.0)


def test_decision_lookup_and_dict():
    d = Decision(
        answers=[Answer("route", "choice", "billing", 0.9, {"billing": 0.9, "other": 0.1})],
        level="L0",
        model="test",
        latency_ms=12.3,
    )
    assert d["route"].value == "billing"
    assert d.get("missing") is None
    payload = d.as_dict()
    assert payload["answers"]["route"]["probability"] == 0.9
