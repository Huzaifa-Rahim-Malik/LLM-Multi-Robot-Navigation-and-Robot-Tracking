#!/usr/bin/env python3

import ollama
import threading


class InferenceConfig:
    """
    Shared CPU/GPU + LLM-model configuration for all Ollama inference calls.
    Thread-safe — GUI and chat loop may access simultaneously.

    num_gpu:
      0  = CPU only  (safe on any machine)
      99 = offload all layers to GPU (requires CUDA + enough VRAM)

    model:
      Ollama model tag used for every generate() call (chat responses AND
      location extraction fallback). Selectable at runtime from the GUI's
      LLM dropdown. Defaults to the original "llama3.2:3b" so behaviour is
      unchanged unless the user explicitly picks a different model.
    """

    # Models offered in the GUI dropdown, each with its approximate,
    # publicly-reported training-data cutoff. Shown to the user after every
    # switch so they know how current that model's world knowledge is.
    # NOTE: several open-weight vendors (Mistral, Qwen) have never published
    # an exact cutoff date — those are marked "unofficial estimate".
    MODEL_CATALOG = {
        "llama3.2:3b": {
            "label":  "Llama 3.2 (3B) — fast, general",
            "cutoff": "December 2023",
            "note":   "Default. Lightweight and quick — good for everyday chit-chat.",
        },
        "llama3.1:8b": {
            "label":  "Llama 3.1 (8B) — stronger general",
            "cutoff": "December 2023",
            "note":   "Bigger Llama — more reliable general-purpose answers, a bit slower.",
        },
        "qwen2.5:7b": {
            "label":  "Qwen 2.5 (7B) — specific / factual",
            "cutoff": "~October 2023 (unofficial estimate)",
            "note":   "Strong on precise/factual questions and multilingual input.",
        },
        "mistral:7b": {
            "label":  "Mistral (7B) — balanced / fast",
            "cutoff": "Not officially published (~2023, community estimate)",
            "note":   "Good all-round balance of speed and answer quality.",
        },
    }

    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(self):
        self._lock   = threading.Lock()
        self.use_gpu = False
        self._model  = self.DEFAULT_MODEL

    @property
    def num_gpu(self) -> int:
        with self._lock:
            return 99 if self.use_gpu else 0

    def set_gpu(self, enabled: bool):
        with self._lock:
            self.use_gpu = enabled

    @property
    def model(self) -> str:
        """Ollama model tag currently selected (thread-safe read)."""
        with self._lock:
            return self._model

    def set_model(self, model_tag: str):
        """Switch the active Ollama model tag (thread-safe write)."""
        with self._lock:
            self._model = model_tag

    @classmethod
    def get_model_info(cls, model_tag: str) -> dict:
        """Catalog metadata (label/cutoff/note) for a model tag, with a safe fallback
        for tags that aren't in MODEL_CATALOG (e.g. a custom pulled model)."""
        return cls.MODEL_CATALOG.get(model_tag, {
            "label":  model_tag,
            "cutoff": "unknown (not in local catalog)",
            "note":   "",
        })

    @staticmethod
    def detect_gpu() -> bool:
        """Return True if an NVIDIA GPU is accessible via nvidia-smi."""
        import subprocess
        try:
            r = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def detect_installed_models() -> list[str]:
        """
        Return the list of Ollama model tags actually pulled/installed on this
        machine (e.g. ['llama3.2:3b', 'qwen2.5:7b']). A model only shows up in
        the GUI dropdown if it appears in this list — nothing that hasn't been
        `ollama pull`-ed is offered as a choice.
        Returns [] if Ollama isn't running / reachable.
        """
        # Try the ollama python client first (same library used for generate()).
        try:
            resp = ollama.list()
            models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
            tags = []
            for m in models:
                name = None
                if isinstance(m, dict):
                    name = m.get("model") or m.get("name")
                else:
                    name = getattr(m, "model", None) or getattr(m, "name", None)
                if name:
                    tags.append(name)
            if tags:
                return tags
        except Exception:
            pass

        # Fallback: shell out to `ollama list` and parse the first column.
        import subprocess
        try:
            r = subprocess.run(["ollama", "list"], capture_output=True,
                                text=True, timeout=8)
            if r.returncode != 0:
                return []
            lines = r.stdout.strip().splitlines()[1:]  # skip header row
            return [ln.split()[0] for ln in lines if ln.strip()]
        except Exception:
            return []


class ResponseGenerator:
    """Generates natural language responses using Ollama with smart conversation history."""

    _NEW_TOPIC_STARTERS = (
        'who is', 'what is', 'where is', 'when is', 'why is', 'how is',
        'tell me about', 'explain',
    )
    _TOPIC_KEYWORDS = frozenset(['captain', 'team', 'president', 'minister', 'director'])

    def __init__(self, logger, location_manager, model: str = "llama3.2:3b",
                 inference_config: InferenceConfig | None = None):
        self.logger           = logger
        self.location_manager = location_manager
        self.model            = model
        self.inference_config = inference_config or InferenceConfig()

    # ── History helpers ───────────────────────────────────────────────────

    def _is_topic_change(self, user_input: str, history) -> bool:
        if not history:
            return False
        lower = user_input.lower().strip()
        if lower.startswith(('no ', 'not ', 'actually ', 'i meant ', "that's wrong")):
            return True
        if any(lower.startswith(s) for s in self._NEW_TOPIC_STARTERS):
            return True
        if any(w in lower for w in self._TOPIC_KEYWORDS):
            return True
        return False

    def _relevant_history(self, user_input: str, history) -> list:
        if not history:
            return []
        items = list(history)
        if self._is_topic_change(user_input, items):
            return items[-1:]
        return items[-3:]

    # ── Quick template responses (no LLM needed) ─────────────────────────

    def generate_response(self, user_input: str, intent=None, location=None,
                          status=None, available_locations=None,
                          conversation_history=None) -> str:

        # Fast-path template responses
        if location and status == "navigating":
            return f"Heading to {location} now!"
        if location and status == "arrived":
            return f"I've arrived at {location}!"
        if status == "unknown":
            locs = ", ".join(self.location_manager.get_all_locations())
            prompt = (
                "You are a friendly indoor robot assistant. "
                f'The user said: "{user_input}". '
                "This does not match any known location on your map. "
                "Politely say you don't recognise it, then list the available "
                "locations naturally. Keep your reply under 60 words. "
                f"Available locations: {locs}."
            )
            try:
                print("__GUI_GENERATING__")
                return ollama.generate(
                    model=self.inference_config.model, prompt=prompt,
                    options={"num_gpu": self.inference_config.num_gpu}
                )["response"].strip()
            except Exception as e:
                self.logger.error(f"Response generation error: {e}")
                return f"I don't recognise that location. Available places: {locs}"

        # General conversational response via Ollama
        history_block = ""
        if conversation_history:
            relevant = self._relevant_history(user_input, conversation_history)
            if relevant:
                history_block = "\n\nRecent conversation:\n" + "".join(
                    f"User: {h['user']}\nRobot: {h['bot']}\n" for h in relevant
                )

        prompt = (
            "You are a friendly, helpful indoor robot assistant. "
            "Give a direct, warm response in 50-60 words. "
            "Answer the current question — don't force links to unrelated prior topics."
            f"{history_block}\n\nUser: {user_input}\n\nRobot:"
        )

        try:
            print("__GUI_GENERATING__")
            return ollama.generate(
                model=self.inference_config.model, prompt=prompt,
                options={"num_gpu": self.inference_config.num_gpu}
            )['response'].strip()
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            return ("I'm ready to help! Ask me anything about navigation, "
                    "my sensors, or just chat.")