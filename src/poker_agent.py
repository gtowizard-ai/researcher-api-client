import random
from typing import Protocol

from models import ActRequest, GameServiceResponse


class PokerAgent(Protocol):
    async def act(self, game_state: GameServiceResponse) -> ActRequest: ...


class CheckCallAgent:
    async def act(self, game_state: GameServiceResponse) -> ActRequest:
        legal_actions = game_state.game_state.legal_actions
        if "k" in legal_actions:
            return ActRequest(action="k")
        if "c" in legal_actions:
            return ActRequest(action="c")
        return ActRequest(action=legal_actions[0])


class AllinAgent:
    async def act(self, game_state: GameServiceResponse) -> ActRequest:
        legal_actions = game_state.game_state.legal_actions
        if "b" in legal_actions:
            raise_range = game_state.game_state.raise_range
            if raise_range is not None:
                return ActRequest(action="b", amount=int(raise_range.max))
        if "c" in legal_actions:
            return ActRequest(action="c")
        return ActRequest(action=legal_actions[0])


class RandomUniformAgent:
    async def act(self, game_state: GameServiceResponse) -> ActRequest:
        legal_actions = game_state.game_state.legal_actions
        sampled_action = random.choice(legal_actions)
        if sampled_action == "b":
            raise_range = game_state.game_state.raise_range
            if raise_range is None:
                sampled_action = "c" if "c" in legal_actions else legal_actions[0]
            else:
                return ActRequest(action="b", amount=int(random.uniform(raise_range.min, raise_range.max)))
        return ActRequest(action=sampled_action)


class AlwaysFoldAgent:
    async def act(self, game_state: GameServiceResponse) -> ActRequest:
        legal_actions = game_state.game_state.legal_actions
        if "f" in legal_actions:
            return ActRequest(action="f")
        if "k" in legal_actions:
            return ActRequest(action="k")
        return ActRequest(action=legal_actions[0])
