#!/usr/bin/env python3

class IntentDetector:
    """Detects user intent — navigate or chat — using fast rule-based matching."""

    _NAVIGATE_TRIGGERS = frozenset([
        'go to', 'take me to', 'navigate to', 'move to', 'head to',
        'drive to', 'travel to', 'get to', 'reach', 'visit',
        'i want to go', "let's go to", 'bring me to',
    ])

    _CHAT_PREFIXES = (
        'what', 'why', 'how', 'when', 'where', 'which', 'who',
        'can you', 'could you', 'would you', 'will you',
        'do you', 'are you', 'is it', 'is the', 'is this',
        'tell me', 'show me', 'explain',
        "no ", "not ", "don't", "won't", "can't", "didn't",
        'never', 'nothing', 'none',
        'where am i', 'where are you', 'current location',
        'current position', 'your position', 'my position',
        'where is the robot', 'robot location', 'starting position',
        'previous position', 'last position',
        'what sensor', 'which sensor', 'what is',
        'tell me about', 'information about',
        'you are not', "you're not", "that's wrong",
        'incorrect', "you're wrong",
        'yes', 'yeah', 'yep', 'ok', 'okay', 'sure',
        'thanks', 'thank you', 'nope', 'nah',
    )

    def __init__(self, logger):
        self.logger = logger

    def detect_intent(self, user_input: str) -> str:
        lower = user_input.lower().strip()

        # Navigation: explicit trigger phrase anywhere in input
        if any(t in lower for t in self._NAVIGATE_TRIGGERS):
            self.logger.info("Intent: NAVIGATE (rule-based)")
            return 'navigate'

        # Chat: input starts with a known chat prefix
        if any(lower.startswith(p) or f' {p}' in lower for p in self._CHAT_PREFIXES):
            self.logger.info("Intent: CHAT (rule-based)")
            return 'chat'

        self.logger.info("Intent: UNCERTAIN")
        return 'uncertain'