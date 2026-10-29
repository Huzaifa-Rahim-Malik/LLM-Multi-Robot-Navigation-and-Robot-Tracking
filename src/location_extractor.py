#!/usr/bin/env python3

import re
import ollama
from response_generator import InferenceConfig


class LocationExtractor:
    """Extracts navigation targets from user input (coordinates, named locations, or AI fallback)."""

    MAP_BOUNDS = {'min_x': 0, 'max_x': 8, 'min_y': 0, 'max_y': 3}

    # Synonym map: canonical name → list of alternate phrasings
    SYNONYMS: dict[str, list[str]] = {
        'hr office':          ['hr', 'hr department', 'hr dept', 'human resources',
                               'human resources dept', 'human resources office'],
        'admin office':       ['admin', 'administration', 'admin dept',
                               'administration dept', 'administration office'],
        'director office':    ["director", "director's office", 'directors office'],
        'dg office':          ['dg', 'director general', "dg's office",
                               'director general office',
                               # Whisper mishearings of "D G" initialism:
                               'digi', 'digi office', 'digie', 'digie office',
                               'digital office', 'd.g.', 'd g', 'd g office',
                               'director generals office'],
        'ai&ds lab':          ['ai lab', 'ds lab', 'ai and ds lab',
                               'artificial intelligence lab', 'data science lab'],
        'pa&i lab':           ['pa lab', 'pai lab', 'pa and i lab', 'pa&i'],
        'cv lab':             ['computer vision lab', 'cv', 'vision lab', 'comp vision lab'],
        'quantum lab':        ['quantum', 'quantum physics lab', 'quantum physics'],
        'caad office':        ['caad', 'caad dept', 'caad department'],
        'procurement office': ['procurement', 'procurement dept', 'procurement department'],
        # BOQ / MOQ — Whisper mishears spoken letter sequences
        'boq':                ['b o q', 'b.o.q', 'b.o.q.', 'boc', 'bokeh',
                               'bo queue', 'b o cue'],
        'moq':                ['m o q', 'm.o.q', 'm.o.q.', 'moc',
                               'mo queue', 'm o cue'],
        'physics':            ['physics lab', 'physics dept', 'physics department'],
    }

    # Pre-build reverse map: synonym → canonical
    _REVERSE: dict[str, str] = {
        syn: canon for canon, syns in SYNONYMS.items() for syn in syns
    }

    def __init__(self, logger, location_manager, model: str = "llama3.2:3b",
                 inference_config=None):
        self.logger           = logger
        self.location_manager = location_manager
        self.model            = model
        self.inference_config = inference_config or InferenceConfig()

    # ── Coordinate extraction ─────────────────────────────────────────────

    def extract_coordinates(self, user_input: str) -> dict | None:
        pattern = r'\(?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)?'
        m = re.search(pattern, user_input)
        if not m:
            return None
        x, y = float(m.group(1)), float(m.group(2))
        b = self.MAP_BOUNDS
        if b['min_x'] <= x <= b['max_x'] and b['min_y'] <= y <= b['max_y']:
            return {'x': x, 'y': y, 'theta': 0.0, 'is_coordinate': True}
        self.logger.warn(f"Coordinates ({x}, {y}) outside map bounds")
        return None

    # ── Named location matching ───────────────────────────────────────────

    def find_matching_locations(self, user_input: str) -> list[str]:
        lower = user_input.lower().strip()
        all_locs = self.location_manager.get_all_locations()

        # Exact → partial against known location names
        # Partial rule: location tokens must cover ≥ 75 % of the meaningful input
        # tokens so "dg office" (2 tokens) does NOT match "dg wd office" (3 tokens,
        # 67 %) but DOES match "go to dg office" after stripping nav trigger words.
        _NAV_TRIGGERS = frozenset([
            'go', 'to', 'take', 'me', 'navigate', 'move', 'head',
            'drive', 'travel', 'get', 'reach', 'visit', 'bring',
        ])
        input_tokens = [t for t in lower.split() if t not in _NAV_TRIGGERS]

        exact, partial = [], []
        for loc in all_locs:
            loc_l = loc.lower()
            if lower == loc_l:
                return [loc]  # fast-exit on exact match

            loc_tokens = loc_l.split()

            # Case A: user input is contained within the location name
            # e.g. user says "admin" → matches "Admin Office"
            if lower in loc_l:
                partial.append(loc)
                continue

            # Case B: location tokens are a subset of the meaningful input tokens
            # Require ≥ 75 % coverage to prevent "dg office" matching "dg wd office"
            # (2 out of 3 meaningful tokens = 67 % → rejected)
            if input_tokens:
                matched = sum(1 for t in loc_tokens if t in input_tokens)
                coverage = matched / len(input_tokens)
                if matched == len(loc_tokens) and coverage >= 0.75:
                    partial.append(loc)

        if partial:
            return partial

        # Synonym fallback
        canon = self._REVERSE.get(lower)
        if canon:
            for loc in all_locs:
                if loc.lower() == canon:
                    return [loc]

        return []

    def extract_location_simple(self, user_input: str) -> str | list[str] | None:
        matches = self.find_matching_locations(user_input)
        if len(matches) == 1:
            return matches[0]
        return matches or None

    # ── Disambiguation ────────────────────────────────────────────────────

    def disambiguate_location(self, matches: list[str]) -> str | None:
        print("\n🤔 Multiple locations match your input:")
        print("─" * 40)
        for i, loc in enumerate(matches, 1):
            d = self.location_manager.get_location(loc)
            print(f"  {i}. {loc}\n     ↳ Position: ({d['x']:.2f}, {d['y']:.2f})")
        print("─" * 40)
        print("  0. Cancel navigation\n")

        while True:
            try:
                choice = input(f"Select option (0-{len(matches)}): ").strip()
                if choice == '0':
                    print("Robot: Navigation cancelled.")
                    return None
                n = int(choice)
                if 1 <= n <= len(matches):
                    selected = matches[n - 1]
                    print(f"✓ Selected: {selected}\n")
                    return selected
                print(f"⚠  Enter a number between 0 and {len(matches)}")
            except ValueError:
                print("⚠  Please enter a valid number")
            except KeyboardInterrupt:
                print("\nRobot: Navigation cancelled.")
                return None

    # ── AI fallback ───────────────────────────────────────────────────────

    # Words that are never part of a location name and should be ignored
    # when deciding whether an input contains "extra" tokens.
    _AI_SKIP_WORDS = frozenset([
        # navigation triggers
        'go', 'to', 'take', 'me', 'navigate', 'move', 'head',
        'drive', 'travel', 'get', 'reach', 'visit', 'bring',
        'i', 'want', 'let', "let's", 'please', 'can', 'you',
        # common filler
        'the', 'a', 'an', 'of', 'and', 'or', 'in', 'at', 'there',
        'now', 'quickly', 'back', 'again', 'just', 'also', 'some',
    ])

    def extract_location_with_ai(self, user_input: str) -> str | None:
        """
        AI fallback for location extraction.

        Guard: if the input contains tokens that are NOT present in any known
        location name (after stripping nav/filler words), the user almost
        certainly named a non-existent place — skip the LLM entirely so it
        cannot hallucinate a nearest-match (e.g. "DG WD office" → "DG Office").
        """
        all_locs = self.location_manager.get_all_locations()

        # Meaningful tokens from the user input (exclude nav/filler words).
        input_tokens = [
            t for t in user_input.lower().split()
            if t not in self._AI_SKIP_WORDS
        ]

        # Guard: every meaningful input token must fit inside the SAME single
        # known location.  Checking each token against the union of all location
        # tokens is insufficient — "DG canteen" passes that check because
        # 'dg' ∈ "DG Office" and 'canteen' ∈ "Canteen", yet the combination
        # does not name any real location.
        #
        # Require that at least one location's token set is a superset of all
        # meaningful input tokens.  If none exists, the user described a place
        # that is not on the map; the LLM must not be allowed to pick the
        # nearest-sounding match.
        if input_tokens:
            single_loc_covers_all = any(
                all(t in set(loc.lower().split()) for t in input_tokens)
                for loc in all_locs
            )
            if not single_loc_covers_all:
                self.logger.info(
                    f"AI fallback skipped — no single location covers all tokens "
                    f"{input_tokens}; input likely names a non-existent place."
                )
                return None

        locations_str = ", ".join(all_locs)
        prompt = (
            f"Extract the location name from this navigation command.\n\n"
            f"Available locations (EXACT names only):\n{locations_str}\n\n"
            f'User command: "{user_input}"\n\n'
            f"Return ONLY the exact location name from the list above, "
            f"or NONE if no location matches. No explanation."
        )
        try:
            raw = ollama.generate(model=self.inference_config.model, prompt=prompt, options={'num_gpu': self.inference_config.num_gpu})['response'].strip()
            self.logger.info(f"AI extracted: '{raw}'")
            clean = raw.lower().strip('"\'.,!? ')
            if clean == 'none':
                return None
            for loc in all_locs:
                if loc.lower() == clean:
                    return loc
            for loc in all_locs:
                if loc.lower() in clean or clean in loc.lower():
                    return loc
        except Exception as e:
            self.logger.error(f"Ollama error: {e}")
        return None

    # ── Main entry point ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_asr(text: str) -> str:
        """
        Strip punctuation that ASR engines (e.g. Whisper) append to words.
        Examples:  "Go to Lawn."              -> "Go to Lawn"
                   "Go to Predictive Lab."    -> "Go to Predictive Lab"
        Decimal digits are left untouched so coordinates like "(3.0, 1.5)"
        pass through unmodified.
        """
        import re
        # Remove punctuation that immediately follows a letter (not a digit).
        text = re.sub(r'(?<=[a-zA-Z])[.!?,;:]+', ' ', text)
        return ' '.join(text.split())   # collapse any extra whitespace

    def extract_location(self, user_input: str):
        # Normalise ASR output before any matching or tokenisation.
        user_input = self._normalize_asr(user_input)

        # 1. Coordinates
        coords = self.extract_coordinates(user_input)
        if coords:
            self.logger.info(f"Extracted coordinates: ({coords['x']}, {coords['y']})")
            return coords

        # 2. Simple name match
        result = self.extract_location_simple(user_input)
        if isinstance(result, str):
            self.logger.info(f"Quick match: {result}")
            return result
        if isinstance(result, list):
            self.logger.info(f"Multiple matches: {result}")
            return self.disambiguate_location(result)

        # 3. AI fallback
        return self.extract_location_with_ai(user_input)