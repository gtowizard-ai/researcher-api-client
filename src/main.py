import asyncio
import time
from http import HTTPStatus

import httpx
import structlog
from fire import Fire
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

from poker_agent import AllinAgent, CheckCallAgent, PokerAgent

_DEFAULT_GAME_NAME = "HUNL 200BB"
_API_URL = "https://researcher.gtowizard.com"
# We limit the number of concurrent hands to 5
_MAX_CONCURRENT_HANDS = 5
_AGENTS = {
    "allin": AllinAgent,
    "checkcall": CheckCallAgent,
}
logger = structlog.get_logger(__name__)


def _is_engine_busy_exception(exception: Exception) -> bool:
    """
    502, 503, and 504 errors can happen under high concurrency
    """
    return isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code in (
        HTTPStatus.BAD_GATEWAY,
        HTTPStatus.SERVICE_UNAVAILABLE,
        HTTPStatus.GATEWAY_TIMEOUT,
    )


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    exception = retry_state.outcome.exception()
    wait_time = retry_state.next_action.sleep
    hand_id = retry_state.kwargs.get("hand_id")
    if isinstance(exception, httpx.HTTPStatusError):
        error_msg = f"{exception.response.status_code} {exception.response.reason_phrase}"
    else:
        error_msg = str(exception)
    logger.debug(
        f"Engine busy. Waiting {wait_time:.2f}s",
        extra={"hand_id": hand_id, "attempt": retry_state.attempt_number, "error": error_msg},
    )


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

    @retry(
        retry=retry_if_exception(_is_engine_busy_exception),
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=2, min=2, max=15),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    async def _post_with_retry(self, url: str, json_data: dict, hand_id: int | None = None) -> httpx.Response:
        response = await self._client.post(url, json=json_data)
        response.raise_for_status()
        return response

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


async def main(api_key: str, agent: str = "allin", num_hands: int = 1000):
    """
    Play `num_hands` hands against GTO Wizard AI.
    Args:
         api_key: User API Key to the Researcher API
         agent: The poker agent to use
         num_hands: Total hands to be played
    """
    agent_class = _AGENTS.get(agent.lower())
    if agent_class is None:
        raise ValueError(f"Unknown agent: {agent}. Available: {', '.join(_AGENTS.keys())}")
    agent = agent_class()
    async with AgentRunner.from_config(agent, api_key) as runner:
        await runner.run(num_hands)


if __name__ == "__main__":
    Fire(lambda **kwargs: asyncio.run(main(**kwargs)))
