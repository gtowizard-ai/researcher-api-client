from typing import Protocol


class PokerAgent(Protocol):
    async def act(self, response: dict) -> dict: ...


class CheckCallAgent:
    async def act(self, response: dict) -> dict:
        legal_actions = response["game_state"]["legal_actions"]
        action = "k" if "k" in legal_actions else "c"
        return {"action": action}


class AllinAgent:
    async def act(self, response: dict) -> dict:
        legal_actions = response["game_state"]["legal_actions"]
        if "b" in legal_actions:
            action = "b"
            amount = response["game_state"]["raise_range"]["max"]
        else:
            action = "c"
            amount = None
        return {"action": action, "amount": amount}
