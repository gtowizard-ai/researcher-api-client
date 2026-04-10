"""Tests for poker agent safety and correctness."""

import asyncio

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import ActionRange, GameModel, GameServiceResponse, GameState, Player
from poker_agent import AllinAgent, AlwaysFoldAgent, CheckCallAgent, RandomUniformAgent


def _make_response(legal_actions: list[str], raise_range: ActionRange | None = None) -> GameServiceResponse:
    """Helper to build a GameServiceResponse with the given legal actions and raise range."""
    return GameServiceResponse(
        hand_id=1,
        game=GameModel(
            game_id=1,
            game_name="HUNL 200BB",
            game_format="cash",
            starting_stack=200.0,
            blinds=[0.5, 1.0],
            stack_reset_per_hand=True,
        ),
        game_state=GameState(
            street="flop",
            common_pot=10.0,
            total_pot=10.0,
            board_cards="AhKdQs",
            is_hand_over=False,
            players=[
                Player(name="hero", stack=195.0, position="BTN", hole_cards="AsAd"),
                Player(name="villain", stack=195.0, position="BB", hole_cards=None),
            ],
            legal_actions=legal_actions,
            raise_range=raise_range,
            action_history=["c", "b2"],
            has_gto_wizard_folded=False,
            winnings=None,
            aivat_score=None,
        ),
    )


# ── AllinAgent ──────────────────────────────────────────────────


class TestAllinAgent:
    agent = AllinAgent()

    def test_bets_max_when_raise_range_available(self):
        resp = _make_response(["f", "c", "b"], ActionRange(min=4.0, max=200.0))
        result = asyncio.run(self.agent.act(resp))
        assert result.action == "b"
        assert result.amount == 200

    def test_amount_is_int_not_float(self):
        resp = _make_response(["f", "c", "b"], ActionRange(min=4.0, max=150.5))
        result = asyncio.run(self.agent.act(resp))
        assert isinstance(result.amount, int)

    def test_no_crash_when_raise_range_is_none(self):
        resp = _make_response(["f", "c", "b"], raise_range=None)
        result = asyncio.run(self.agent.act(resp))
        assert result.action in ("c", "f", "b")

    def test_calls_when_bet_not_available(self):
        resp = _make_response(["f", "c"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action == "c"

    def test_action_always_in_legal_actions(self):
        resp = _make_response(["f"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action in ["f"]


# ── CheckCallAgent ──────────────────────────────────────────────


class TestCheckCallAgent:
    agent = CheckCallAgent()

    def test_checks_when_available(self):
        resp = _make_response(["k", "b"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action == "k"

    def test_calls_when_check_unavailable(self):
        resp = _make_response(["f", "c", "b"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action == "c"

    def test_no_illegal_action_when_only_fold_and_bet(self):
        resp = _make_response(["f", "b"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action in ["f", "b"]

    def test_single_legal_action(self):
        resp = _make_response(["k"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action == "k"


# ── RandomUniformAgent ──────────────────────────────────────────


class TestRandomUniformAgent:
    agent = RandomUniformAgent()

    def test_no_crash_when_raise_range_is_none(self):
        resp = _make_response(["f", "c", "b"], raise_range=None)
        for _ in range(20):
            result = asyncio.run(self.agent.act(resp))
            assert result.action in ["f", "c", "b"]

    def test_bet_amount_is_int(self):
        resp = _make_response(["b"], ActionRange(min=4.0, max=200.0))
        result = asyncio.run(self.agent.act(resp))
        if result.action == "b":
            assert isinstance(result.amount, int)

    def test_bet_amount_within_range(self):
        rr = ActionRange(min=10.0, max=50.0)
        resp = _make_response(["f", "c", "b"], rr)
        for _ in range(20):
            result = asyncio.run(self.agent.act(resp))
            if result.action == "b":
                assert 10 <= result.amount <= 50

    def test_action_always_legal(self):
        resp = _make_response(["f", "c"])
        for _ in range(20):
            result = asyncio.run(self.agent.act(resp))
            assert result.action in ["f", "c"]

    def test_single_value_range(self):
        resp = _make_response(["b", "c"], ActionRange(min=100.0, max=100.0))
        result = asyncio.run(self.agent.act(resp))
        if result.action == "b":
            assert result.amount == 100


# ── AlwaysFoldAgent ─────────────────────────────────────────────


class TestAlwaysFoldAgent:
    agent = AlwaysFoldAgent()

    def test_folds_when_available(self):
        resp = _make_response(["f", "c", "b"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action == "f"

    def test_no_illegal_action_when_fold_unavailable(self):
        resp = _make_response(["c", "b"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action in ["c", "b"]

    def test_checks_when_only_check_available(self):
        resp = _make_response(["k", "b"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action == "k"

    def test_single_legal_action(self):
        resp = _make_response(["c"])
        result = asyncio.run(self.agent.act(resp))
        assert result.action == "c"
