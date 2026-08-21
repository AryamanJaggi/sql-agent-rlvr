"""A scripted stand-in for a real model. Used to test env/ without 
the computational cost of loading and calling the real model.
"""

from __future__ import annotations


class ScriptedPolicy:
    """Returns the next line from a fixed script. Repeats the last entry
    forever once the script runs out. Useful for step-cap-exhaustion
    tests, where we want a policy that never calls final_answer no
    matter how long the episode runs.

    Position in the script comes from the transcript itself (count of
    "Observation:" markers = completed turns so far), not a persistent
    call counter. A fresh episode always starts at zero observations, so
    the same instance can get reused across many episodes without needing 
    a new one each time.
    """

    def __init__(self, script: list[str]):
        if not script:
            raise ValueError("script must be non-empty")
        self.script = script

    def __call__(self, transcript: str) -> str:
        turn = transcript.count("Observation:")
        index = min(turn, len(self.script) - 1)
        return self.script[index]