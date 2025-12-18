# GTO Wizard AI Researcher API Client
Despite the growing interest in applying generalist AI agents and LLMs to games, a standardized platform for benchmarking their performance in poker has been lacking. We introduce GTO Wizard Research API: a public API that provides a standardized environment for benchmarking AI agents in No-Limit Texas Hold 'em, the most popular variant of poker.

This API enables researchers to test their agents against [GTO Wizard AI](https://blog.gtowizard.com/introducing-gto-wizard-ai/), a proprietary state-of-the-art poker agent that demonstrated superior performance against past winners of the Annual Computer Poker Competition. Our system evaluates agents using [AIVAT](https://arxiv.org/abs/1612.06915), a provably unbiased low-variance technique for evaluating performance in imperfect information games, reducing the need to play tens of thousands of hands to gather statistically significant results. We make the evaluation results publicly available through a free website and a real-time leaderboard.

## Documentation
To dive deeper on how to benchmark your agent against GTO Wizard AI, check out the [API documentation](https://researcher.gtowizard.com/docs).

## Installation
**Prerequisites:**
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

**Setup:**
```bash
# Install dependencies
uv sync
```

## Usage

This repository provides skeleton agents to benchmark against GTO Wizard AI:
- **allin**: Always goes all-in when possible, otherwise calls
- **checkcall**: Always checks when possible, otherwise calls

**Run an agent:**
```bash
cd src && uv run python -m main --api-key YOUR_API_KEY --agent allin --hands 10
```
