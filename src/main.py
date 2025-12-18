import asyncio
import random
import time
from http import HTTPStatus

import httpx
import structlog
from fire import Fire
from tqdm.asyncio import tqdm

from poker_agent import AllinAgent, CheckCallAgent, PokerAgent

_NUM_HANDS = 1000
_RETRY_BASE_DELAY = 2.0
_RETRY_MAX_DELAY = 15.0
_DEFAULT_GAME_NAME = "HUNL 200BB"
_API_URL = "https://researcher.gtowizard.com"
# We limit the number of concurrent hands to 5
_MAX_CONCURRENT_HANDS = 5
_AGENTS = {
    "allin": AllinAgent,
    "checkcall": CheckCallAgent,
}
logger = structlog.get_logger(__name__)


class AgentRunner:
    def __init__(self, client: httpx.AsyncClient, agent: PokerAgent):
        self._client = client
        self._agent = agent
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_HANDS)

    @classmethod
    def from_config(cls, agent: PokerAgent, api_key: str) -> "AgentRunner":
        limits = httpx.Limits(
            max_keepalive_connections=_MAX_CONCURRENT_HANDS,
            max_connections=_MAX_CONCURRENT_HANDS * 2,
        )
        client = httpx.AsyncClient(
            base_url=_API_URL,
            headers={"X-API-KEY": api_key},
            timeout=180,
            limits=limits,
        )
        return cls(client, agent)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()

    # 503 errors are expected under high concurrency; retry with exponential backoff
    async def _post_with_retry(
        self,
        url: str,
        json_data: dict,
        max_retries: int = 20,
        hand_id: int | None = None,
    ) -> httpx.Response:
        for attempt in range(max_retries):
            try:
                response = await self._client.post(url, json=json_data)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code != HTTPStatus.SERVICE_UNAVAILABLE or attempt == max_retries - 1:
                    raise
            delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
            sleep_time = random.uniform(1.0, delay)
            logger.debug(
                f"503 error. Waiting {sleep_time:.2f}s",
                extra={"attempt": attempt + 1, "hand_id": hand_id},
            )
            await asyncio.sleep(sleep_time)
        raise RuntimeError("Unreachable")

    async def _create_new_hand(self) -> dict:
        request = {"game_name": _DEFAULT_GAME_NAME}
        response = await self._post_with_retry("/hands", json_data=request)
        return response.json()

    async def _act(self, hand_id: int, game_state: dict) -> dict:
        action_request = await self._agent.act(game_state)
        response = await self._post_with_retry(f"/hands/{hand_id}/act", json_data=action_request, hand_id=hand_id)
        return response.json()

    async def _play_hand(self) -> bool:
        hand_id = None
        async with self._semaphore:
            try:
                response = await self._create_new_hand()
                hand_id = response["hand_id"]

                while not response["game_state"]["is_hand_over"]:
                    response = await self._act(hand_id, response)
                return True
            except httpx.HTTPStatusError as e:
                logger.error(f"API Error: {e.response.text}", extra={"hand_id": hand_id})
                return False
            except Exception as e:
                logger.error(f"Unexpected error: {e}", extra={"hand_id": hand_id})
                return False

    async def run(self, num_hands: int) -> None:
        logger.info(f"Starting {num_hands} hands on game {_DEFAULT_GAME_NAME}")
        start_time = time.time()
        hands = [self._play_hand() for _ in range(num_hands)]
        results = []
        for hand in tqdm.as_completed(hands, total=num_hands, desc="Playing hands"):
            result = await hand
            results.append(result)

        end_time = time.time()
        duration = end_time - start_time
        successful_hands = sum(results)
        failed_hands = len(results) - successful_hands
        seconds_per_hand = duration / num_hands if num_hands > 0 else 0
        logger.info("Benchmark finished")
        logger.info(
            f"Successful hands: {successful_hands}. Failed hands: {failed_hands}. Average seconds/hand: {seconds_per_hand:.3f}"
        )


async def main(api_key: str, agent: str = "allin", hands: int = _NUM_HANDS):
    """
    Plays N hands against GTO Wizard AI.
    Args:
         api_key: User API Key to the Researcher API
         agent: The poker agent to use
         hands: Total hands to be played
    """
    agent_class = _AGENTS.get(agent.lower())
    if agent_class is None:
        raise ValueError(f"Unknown agent: {agent}. Available: {', '.join(_AGENTS.keys())}")
    agent = agent_class()
    async with AgentRunner.from_config(agent, api_key) as runner:
        await runner.run(hands)


if __name__ == "__main__":
    Fire(lambda **kwargs: asyncio.run(main(**kwargs)))
