#!/usr/bin/env python
"""
CLI entry point for running crews locally (dev/testing).

The primary way to run the full multi-crew workflow is via `streamlit run app.py`.
This file is useful for testing individual crews in isolation.
"""

import sys
import json
from fast_auction_research___speed_optimized.crew import (
    ScreeningCrewPartA,
    ScreeningCrewPartB,
    PerLotValidationCrew,
    PerLotDeepResearchCrew,
    SynthesisCrew,
)


def run():
    """
    Run Phase 1a (catalog extraction) as a quick smoke test.
    """
    inputs = {
        "auction_url": "sample_value",
        "platform_fee_choice": "3_percent_surcharge",
    }
    result = ScreeningCrewPartA().crew().kickoff(inputs=inputs)
    print(result)


def train():
    """
    Train Phase 1a crew for a given number of iterations.
    """
    inputs = {
        "auction_url": "sample_value",
        "platform_fee_choice": "3_percent_surcharge",
    }
    try:
        ScreeningCrewPartA().crew().train(
            n_iterations=int(sys.argv[2]),
            filename=sys.argv[3],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay a crew execution from a specific task.
    """
    try:
        ScreeningCrewPartA().crew().replay(task_id=sys.argv[2])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test Phase 1a crew execution and return results.
    """
    inputs = {
        "auction_url": "sample_value",
        "platform_fee_choice": "3_percent_surcharge",
    }
    try:
        ScreeningCrewPartA().crew().test(
            n_iterations=int(sys.argv[2]),
            openai_model_name=sys.argv[3],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        print("Commands: run, train, replay, test")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
