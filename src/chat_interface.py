#!/usr/bin/env python3
"""
chat_interface.py — TurtleBot Navigation Assistant
====================================================
Architecture (post-threading refactor):

  TTS thread       — serialises all speech via _tts_queue; never blocks chat loop
  Nav thread       — runs navigate_to_location() in background; chat loop stays live
  Arrival queue    — nav thread posts events; announcements are contextual at speak-time
  Chat loop        — accepts input and LLM calls freely during navigation

Multi-location:  "go to home then lawn"  →  nav_queue = [Home, Lawn]
PosePublisher:   fires at ACTUAL arrival time, independent of when TTS speaks
"""

import re
import math
import queue as _queue
import threading
import time
from collections import deque

from rclpy.node import Node

from intent_detector    import IntentDetector
from location_extractor import LocationExtractor
from response_generator import ResponseGenerator
from robot_navigator    import RobotNavigator
from location_manager   import LocationManager


# Regex that splits "go to home then lawn and then canteen" into parts
_MULTI_SEP = re.compile(
    r'\s+(?:then(?:\s+go\s+to)?|and\s+then(?:\s+go\s+to)?|after\s+that(?:\s+go\s+to)?'
    r'|followed\s+by|next\s+go\s+to)\s+',
    re.IGNORECASE,
)


class ChatInterface:
    def __init__(self, logger, location_manager: LocationManager,
                 location_extractor, response_generator, robot_navigator,
                 intent_detector, node: Node,
                 voice_input=None, voice_output=None,
                 pose_publisher=None):

        self.logger             = logger
        self.node               = node
        self.location_manager   = location_manager
        self.location_extractor = location_extractor
        self.response_generator = response_generator
        self.robot_navigator    = robot_navigator
        self.intent_detector    = intent_detector
        self.pose_publisher     = pose_publisher
        self.voice_input        = voice_input
        self.voice_output       = voice_output

        self.mode     = 'text'
        self.gui_mode = False
        self.gui_mode_changed_callback = None

        # Conversation / navigation history
        self.conversation_history = deque(maxlen=5)
        self.navigation_history   = deque(maxlen=10)

        self.last_response_asked_question = False
        self.pending_navigation     = None
        self.pending_disambiguation = None
        self.current_topic          = None
        self.navigation_cancel_requested = False

        # ── TTS serialisation ─────────────────────────────────────────
        # A single dedicated thread drains _tts_queue so speech never
        # overlaps and the chat loop never blocks waiting for TTS.
        self._tts_queue: _queue.Queue = _queue.Queue()
        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="tts_worker")
        self._tts_thread.start()

        # ── Navigation threading ──────────────────────────────────────
        self._nav_queue: deque        = deque()
        self._nav_lock                = threading.Lock()
        self._nav_active              = threading.Event()   # set while nav thread runs
        self._current_nav_dest        = None                # what is being navigated to now
        self._arrival_queue: _queue.Queue = _queue.Queue() # events posted by nav thread

    # ═══════════════════════════════════════════════════════════════
    #  Mode helpers
    # ═══════════════════════════════════════════════════════════════

    def set_mode(self, mode: str) -> bool:
        if mode not in ('text', 'voice'):
            self.logger.error(f"Invalid mode: {mode}")
            return False
        old = self.mode
        self.mode = mode
        self.logger.info(f"Mode: {old} → {mode}")
        if self.gui_mode_changed_callback and old != mode:
            try:
                self.gui_mode_changed_callback(mode)
            except Exception:
                pass
        return True

    # ═══════════════════════════════════════════════════════════════
    #  Input
    # ═══════════════════════════════════════════════════════════════

    def get_user_input(self) -> str:
        if self.mode == 'voice' and not self.gui_mode:
            return self.get_voice_input()
        return self.get_text_input()

    def get_text_input(self) -> str:
        return input("" if self.gui_mode else "YOU: ").strip()

    def get_voice_input(self) -> str:
        if not self.voice_input:
            self.output_response("Voice not available — switching to text.")
            self.mode = 'text'
            return self.get_text_input()
        try:
            text = self.voice_input.listen(
                auto_stop=True,
                context_locations=self.location_manager.get_all_locations())
            if text:
                print(f"YOU (voice): {text}")
                return text
            return ""
        except Exception as e:
            self.logger.error(f"Voice input error: {e}")
            return ""

    # ═══════════════════════════════════════════════════════════════
    #  TTS output  (print immediately; speech async via worker thread)
    # ═══════════════════════════════════════════════════════════════

    def output_response(self, text: str):
        """Print immediately and queue for TTS — never blocks the caller."""
        print(f"TURTLEBOT: {text}\n")
        if self.voice_output:
            self._tts_queue.put(text)

    def output_text(self, text: str):
        print(f"TURTLEBOT: {text}\n")

    def output_voice(self, text: str):
        """Compatibility shim — routes through the TTS queue like output_response."""
        self.output_response(text)

    def _tts_worker(self):
        """
        Dedicated thread: serialises all TTS so speech never overlaps.

        Items in _tts_queue may be:
          str      — speak immediately (chat responses, confirmations)
          callable — evaluate RIGHT NOW to get the text, then speak.
                     Used for arrival announcements so robot state is
                     checked at the exact moment speech starts, not when
                     the event was first processed.
        """
        while True:
            item = self._tts_queue.get()
            if item is None:           # shutdown sentinel
                break
            try:
                # Evaluate lazy arrival closures at speak-time
                text = item() if callable(item) else item
                if text:
                    if callable(item):
                        # Print here so GUI shows the message at speak-time
                        print(f"TURTLEBOT: {text}\n")
                    self._do_speak_blocking(text)
            except Exception as e:
                self.logger.error(f"TTS worker error: {e}")
            finally:
                self._tts_queue.task_done()

    def _do_speak_blocking(self, text: str):
        """Actually synthesise and play text. Runs only inside _tts_worker."""
        if not self.voice_output:
            return
        self.voice_output.stop_requested = False
        self.voice_output.speaking       = True
        self.voice_output._prepend_silence = True
        try:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            for sentence in sentences:
                if self.voice_output.stop_requested:
                    break
                self.voice_output.speak(sentence, blocking=True)
                time.sleep(0.05)
        except Exception as e:
            self.logger.error(f"TTS speak error: {e}")
        finally:
            self.voice_output.speaking       = False
            self.voice_output.stop_requested = False
            self.voice_output._prepend_silence = True

    # ═══════════════════════════════════════════════════════════════
    #  Navigation threading
    # ═══════════════════════════════════════════════════════════════

    def _start_nav_thread(self):
        """Start background navigation worker if not already running."""
        if self._nav_active.is_set():
            return
        self._nav_active.set()
        threading.Thread(
            target=self._nav_worker, daemon=True, name="nav_worker").start()

    def _nav_worker(self):
        """
        Background thread: pops from _nav_queue and navigates sequentially.
        Posts arrival events to _arrival_queue; does NOT speak anything itself.
        PosePublisher is called here at actual arrival time.
        """
        while True:
            with self._nav_lock:
                if not self._nav_queue:
                    self._nav_active.clear()
                    self._current_nav_dest = None
                    return
                destination = self._nav_queue.popleft()

            self._current_nav_dest = destination
            print(f"__NAV_STARTED__{destination}__")   # picked up by GUI

            if self.pose_publisher:
                self.pose_publisher.set_status("navigating", destination)

            # Already there? — post event and skip Nav2
            if self._is_already_at_target(destination):
                if self.pose_publisher:
                    self.pose_publisher.set_status("arrived", destination)
                with self._nav_lock:
                    next_dest = self._nav_queue[0] if self._nav_queue else None
                self._arrival_queue.put({
                    'completed':    destination,
                    'next_dest':    next_dest,
                    'success':      True,
                    'already_there': True,
                })
                print(f"__NAV_ARRIVED__{destination}__")
                continue

            success = self.robot_navigator.navigate_to_location(destination)

            # ── PosePublisher fires at ACTUAL arrival, independent of TTS ──
            if self.pose_publisher:
                self.pose_publisher.set_status("arrived" if success else "idle", destination)

            with self._nav_lock:
                next_dest = self._nav_queue[0] if self._nav_queue else None

            self._arrival_queue.put({
                'completed':    destination,
                'next_dest':    next_dest,
                'success':      success,
                'already_there': False,
            })
            print(f"__NAV_ARRIVED__{destination}__")

    def cancel_navigation(self):
        """Cancel active navigation and clear the queue."""
        with self._nav_lock:
            self._nav_queue.clear()
        self.navigation_cancel_requested = True
        self._nav_active.clear()
        self._current_nav_dest = None
        try:
            self.robot_navigator.navigator.cancelTask()
        except Exception:
            pass

    # ── Multi-location parsing ────────────────────────────────────

    def parse_multi_location(self, user_input: str) -> list:
        """
        Parse 'go to home then lawn' → ['Home', 'Lawn'].
        Returns list with ≥2 entries, or [] if only one (or zero) location found.
        """
        parts = _MULTI_SEP.split(user_input)
        if len(parts) < 2:
            return []
        locations = []
        for part in parts:
            loc = self.location_extractor.extract_location(part.strip())
            if loc and not isinstance(loc, list):
                locations.append(loc)
        return locations if len(locations) >= 2 else []

    # ── Queue a navigation request ────────────────────────────────

    def _queue_navigation(self, locations: list, user_input: str = "") -> str:
        """
        Enqueue one or more destinations, start the nav thread, speak confirmation.
        Returns the spoken confirmation message.
        """
        source_loc = self.show_navigation_context(locations[0])

        with self._nav_lock:
            self._nav_queue.extend(locations)

        if len(locations) == 1:
            dest_str = (
                f"coordinates ({locations[0]['x']}, {locations[0]['y']})"
                if isinstance(locations[0], dict) else str(locations[0])
            )
            msg = f"Heading to {dest_str} now!"
        else:
            names = [
                f"({l['x']}, {l['y']})" if isinstance(l, dict) else str(l)
                for l in locations
            ]
            route = ", ".join(names[:-1]) + f", then {names[-1]}"
            msg = f"Heading to {route}!"

        self.output_response(msg)
        self._start_nav_thread()

        # Start voice cancel listener alongside nav thread (voice mode only)
        self.navigation_cancel_requested = False
        if self.voice_input and self.mode == 'voice' and not self.gui_mode:
            import threading as _nthr
            _nthr.Thread(
                target=self._voice_cancel_navigation_listener,
                daemon=True, name="nav_cancel_listener"
            ).start()

        self.add_to_history(user_input, msg, "navigation", locations[0], source_loc)
        return msg

    # ── Arrival announcement ──────────────────────────────────────

    def _process_arrivals(self):
        """
        Drain _arrival_queue and schedule contextual announcements.

        KEY DESIGN — lazy evaluation at speak-time:
          Instead of building the announcement message now (when the arrival
          event is processed), we capture the events in a closure and push
          that CALLABLE into _tts_queue.  _tts_worker calls it right before
          speaking, so the robot-state check happens at the exact moment TTS
          is about to open its mouth.  This gives the three guaranteed behaviours:

          1. TTS free before robot left  → "I've arrived at Home!"
          2. TTS busy while robot left   → "I've reached Home and I'm now
                                            heading to Lawn."
          3. Both arrivals before TTS    → "I've reached Home and Lawn.
                                            All destinations reached!"
        """
        pending = []
        while True:
            try:
                pending.append(self._arrival_queue.get_nowait())
            except _queue.Empty:
                break

        if not pending:
            return

        # Snapshot captured NOW (process-time).
        # The closure below is evaluated LATER (speak-time).
        events_snapshot = list(pending)

        def _lazy_msg():
            """Called by _tts_worker at the moment speech begins."""
            events = events_snapshot          # arrivals captured at process-time

            if len(events) == 1:
                # Single arrival — full contextual check at speak-time
                return self._build_arrival_msg(events[0])

            # Multiple arrivals piled up (robot was fast / TTS was long).
            # Chain them into one sentence.
            names = [ev['completed'] for ev in events]
            chain = (
                f"{names[0]} and {names[1]}" if len(names) == 2
                else ", ".join(names[:-1]) + f", and {names[-1]}"
            )
            last = events[-1]
            next_dest = last.get('next_dest')

            # At speak-time: is the robot STILL navigating to a next stop?
            if next_dest and self._nav_active.is_set():
                still_there = self._is_already_at_target(last['completed'])
                if still_there:
                    return (f"I've arrived at {chain}! "
                            f"Now heading to {next_dest}.")
                return (f"I've reached {chain} and I'm now heading to "
                        f"{next_dest}.")
            return f"I've reached {chain}. All destinations reached!"

        if self.voice_output:
            # Push the lazy closure — evaluated at speak-time inside _tts_worker
            self._tts_queue.put(_lazy_msg)
        else:
            # No TTS — evaluate immediately and print
            msg = _lazy_msg()
            print(f"TURTLEBOT: {msg}\n")

        # Update navigation history now (timing doesn't matter for history)
        for ev in events_snapshot:
            if ev.get('success') and not ev.get('already_there'):
                self.navigation_history.append({
                    'destination':  ev['completed'],
                    'source':       None,
                    'user_command': '',
                })

    def _build_arrival_msg(self, ev: dict) -> str:
        completed    = ev['completed']
        next_dest    = ev.get('next_dest')
        success      = ev.get('success', True)
        already_there = ev.get('already_there', False)

        if not success:
            base = f"I had trouble getting to {completed}."
            return base + (f" Attempting {next_dest} next." if next_dest else "")

        if already_there:
            base = f"I'm already at {completed}!"
            return base + (f" Now heading to {next_dest}." if next_dest else "")

        # Check robot state at speak-time (not at arrival-time)
        still_there = self._is_already_at_target(completed) if isinstance(completed, str) else False

        if next_dest:
            if still_there:
                return f"I've arrived at {completed}! Now heading to {next_dest}."
            else:
                return f"I've reached {completed} and I'm now heading to {next_dest}."
        else:
            return f"I've arrived at {completed}!"

    # ═══════════════════════════════════════════════════════════════
    #  Already-at-target helper
    # ═══════════════════════════════════════════════════════════════

    def _is_already_at_target(self, location) -> bool:
        current_pos = self.robot_navigator.get_current_position()
        if not current_pos:
            return False
        if isinstance(location, dict) and 'is_coordinate' in location:
            tx, ty = location['x'], location['y']
        elif isinstance(location, str):
            loc_data = self.location_manager.get_location(location)
            if not loc_data:
                return False
            tx, ty = loc_data['x'], loc_data['y']
        else:
            return False
        return math.hypot(current_pos['x'] - tx, current_pos['y'] - ty) < 0.3

    # ═══════════════════════════════════════════════════════════════
    #  Navigation context display
    # ═══════════════════════════════════════════════════════════════

    def show_navigation_context(self, destination) -> str | None:
        current_pos = self.robot_navigator.get_current_position()
        if current_pos:
            nearest_loc, distance = self.robot_navigator.find_nearest_location(current_pos)
            dest_str = (
                f"({destination['x']}, {destination['y']})"
                if isinstance(destination, dict) and 'is_coordinate' in destination
                else destination
            )
            if distance < 0.5:
                print(f"📍 Navigation: {nearest_loc} → {dest_str}")
                return nearest_loc
            print(f"📍 Navigation: ({current_pos['x']:.1f}, {current_pos['y']:.1f}) → {dest_str}")
            return f"coordinates ({current_pos['x']:.1f}, {current_pos['y']:.1f})"
        dest_str = (
            f"({destination['x']}, {destination['y']})"
            if isinstance(destination, dict) and 'is_coordinate' in destination
            else destination
        )
        print(f"📍 Navigation: → {dest_str}")
        return None

    # ═══════════════════════════════════════════════════════════════
    #  History helpers
    # ═══════════════════════════════════════════════════════════════

    def add_to_history(self, user_input, bot_response,
                       action_type=None, location=None, source_location=None):
        if self.conversation_history and self.detect_topic_change(user_input):
            self.logger.info("Topic change — clearing context")
            self.conversation_history.clear()

        self.conversation_history.append({
            'user':            user_input,
            'bot':             bot_response,
            'action':          action_type,
            'location':        location,
            'source_location': source_location,
        })

        if action_type == 'navigation' and location:
            self.navigation_history.append({
                'destination':  location,
                'source':       source_location,
                'user_command': user_input,
            })

        self.last_response_asked_question = bot_response.strip().endswith('?')

    def detect_topic_change(self, user_input: str) -> bool:
        lower = user_input.lower().strip()
        if len(lower) < 3:
            return False
        corrections = ['no ', 'not ', 'actually ', 'i meant ', "that's wrong",
                       'incorrect', "you're wrong", 'nope ', 'wrong ']
        if any(lower.startswith(c) for c in corrections):
            return True
        new_topics = ['who is', 'what is', 'where is', 'when is', 'why is',
                      'how is', 'tell me about', 'explain', 'describe']
        if any(lower.startswith(t) for t in new_topics):
            if self.conversation_history:
                last = list(self.conversation_history)[-1]['user'].lower()
                return any(last.startswith(t) for t in new_topics)
        return False

    def get_last_navigation_source(self) -> str | None:
        return self.navigation_history[-1]['source'] if self.navigation_history else None

    def get_last_navigation_destination(self) -> str | None:
        return self.navigation_history[-1]['destination'] if self.navigation_history else None

    def get_previous_location(self, steps_back: int = 1) -> str | None:
        if len(self.navigation_history) >= steps_back:
            return self.navigation_history[-steps_back]['destination']
        return None

    def get_current_location_name(self) -> str | None:
        pos = self.robot_navigator.get_current_position()
        if pos:
            name, dist = self.robot_navigator.find_nearest_location(pos)
            return name if dist < 0.5 else None
        return None

    # ═══════════════════════════════════════════════════════════════
    #  Distance helpers
    # ═══════════════════════════════════════════════════════════════

    def calculate_distance_and_steps(self, from_pos, to_pos):
        dx = to_pos['x'] - from_pos['x']
        dy = to_pos['y'] - from_pos['y']
        dist = math.sqrt(dx**2 + dy**2)
        return dist, int(dist / 0.5)

    def get_location_info(self, location_name: str) -> dict | None:
        if not self.location_manager.location_exists(location_name):
            return None
        loc = self.location_manager.get_location(location_name)
        info = {'name': location_name, **loc}
        pos = self.robot_navigator.get_current_position()
        if pos:
            info['distance'], info['steps'] = self.calculate_distance_and_steps(pos, loc)
        return info

    # ═══════════════════════════════════════════════════════════════
    #  Confirmation / disambiguation
    # ═══════════════════════════════════════════════════════════════

    def confirm_partial_navigation(self, matches) -> str:
        if isinstance(matches, str):
            matches = [matches]
        self.pending_navigation = matches[0]
        self.output_response(f"Did you mean to navigate to {matches[0]}? (yes / no)")
        return "WAITING_CONFIRMATION"

    def handle_confirmation_response(self, user_input: str) -> bool:
        lower = user_input.lower().strip()
        if lower in ('yes', 'yeah', 'yep', 'yup', 'sure', 'ok', 'okay',
                     'correct', 'right', 'y'):
            if self.pending_navigation:
                location = self.pending_navigation
                self.pending_navigation = None
                if self._is_already_at_target(location):
                    response = f"I'm already at {location}!"
                    self.output_response(response)
                    self.add_to_history(user_input, response, "already_there",
                                        location, None)
                else:
                    self._queue_navigation([location], user_input)
                return True

        elif lower in ('no', 'nope', 'nah', 'not really', 'n'):
            self.pending_navigation = None
            response = "Okay, cancelled. What would you like me to do?"
            self.output_response(response)
            self.add_to_history(user_input, response, "navigation_declined")
            return True

        return False

    def _navigate_to_selected(self, selected_location: str, original_input: str):
        """Navigate to a location chosen during disambiguation."""
        if self._is_already_at_target(selected_location):
            source_loc = self.show_navigation_context(selected_location)
            response = f"I'm already at {selected_location}!"
            self.output_response(response)
            self.add_to_history(original_input, response, "already_there",
                                selected_location, source_loc)
        else:
            self._queue_navigation([selected_location], original_input)

    def handle_partial_location_search(self, search_term: str, original_input: str):
        sl = search_term.lower()
        search_key = 'office' if sl in ('offices', 'office') else \
                     'lab'    if sl in ('labs', 'lab', 'laboratory', 'laboratories') else sl

        matching = [loc for loc in self.location_manager.get_all_locations()
                    if search_key in loc.lower()]

        if not matching:
            response = f"No locations found containing '{search_term}'."
            self.output_response(response)
            self.add_to_history(original_input, response, "no_matches")
            return

        print(f"\n📍 Locations containing '{search_term}':")
        print("─" * 55)
        for i, location in enumerate(matching, 1):
            d = self.location_manager.get_location(location)
            print(f"  {i}. {location:<25} → ({d['x']:.1f}, {d['y']:.1f})")
        print("─" * 55 + "\n  0. Cancel\n")

        prompt = (f"Say or type a number 1–{len(matching)} to navigate, or 0 to cancel.")
        self.output_response(prompt)

        if self.gui_mode:
            self.pending_disambiguation = {'matches': matching, 'original_input': original_input}
            return

        while True:
            sel = self.get_voice_input() if self.mode == 'voice' else input("Selection: ").strip()
            if sel and self._process_disambiguation_selection(sel, matching, original_input) == "done":
                return

    def _process_disambiguation_selection(self, selection: str,
                                          matching_locations: list,
                                          original_input: str) -> str:
        sel = selection.strip().rstrip('.')
        if sel == '0' or sel.lower() in ('cancel', 'zero', 'no'):
            response = "Navigation cancelled."
            self.output_response(response)
            self.add_to_history(original_input, response, "cancelled")
            return "done"

        if sel.isdigit():
            choice = int(sel)
            if 1 <= choice <= len(matching_locations):
                self._navigate_to_selected(matching_locations[choice - 1], original_input)
                return "done"
            self.output_response(f"Enter 1–{len(matching_locations)} or 0 to cancel.")
            return "invalid"

        word_nums = {"one":1,"two":2,"three":3,"four":4,"five":5,
                     "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        if sel.lower() in word_nums:
            choice = word_nums[sel.lower()]
            if 1 <= choice <= len(matching_locations):
                self._navigate_to_selected(matching_locations[choice-1], original_input)
                return "done"

        loc = self.location_extractor.extract_location(sel)
        if loc and loc in matching_locations:
            self._navigate_to_selected(loc, original_input)
            return "done"

        if self.handle_mode_switch_command(sel):
            return "invalid"

        self.output_response(f"Enter a number 1–{len(matching_locations)} or 0 to cancel.")
        return "invalid"

    # ═══════════════════════════════════════════════════════════════
    #  Go back
    # ═══════════════════════════════════════════════════════════════

    def handle_go_back_command(self, user_input: str) -> bool:
        lower = user_input.lower().strip()
        triggers = ['go back', 'take me back', 'head back', 'return',
                    'navigate back', 'bring me back', 'get back']
        if not any(t in lower for t in triggers):
            return False

        last_source = self.get_last_navigation_source()
        if last_source and last_source != "None":
            if self._is_already_at_target(last_source):
                response = f"I'm already at {last_source}!"
                self.output_response(response)
                self.add_to_history(user_input, response, "already_there",
                                    last_source, None)
            else:
                self._queue_navigation([last_source], user_input)
        else:
            response = ("I don't have a previous location to go back to. "
                        "Where would you like me to navigate?")
            self.output_response(response)
            self.add_to_history(user_input, response, "no_go_back")
        return True

    # ═══════════════════════════════════════════════════════════════
    #  Position / location queries
    # ═══════════════════════════════════════════════════════════════

    def handle_position_query(self, user_input: str = "where am i"):
        pos  = self.robot_navigator.get_current_position()
        ori  = self.robot_navigator.get_current_orientation()
        if pos:
            nearest, dist = self.robot_navigator.find_nearest_location(pos)
            if nearest and dist < 0.5:
                response = (
                    f"I'm at {nearest}, facing {ori['direction']}. "
                    f"Coordinates: ({pos['x']:.2f}, {pos['y']:.2f})."
                    if ori else
                    f"I'm at {nearest}. Coordinates: ({pos['x']:.2f}, {pos['y']:.2f})."
                )
            else:
                response = (
                    f"I'm at ({pos['x']:.2f}, {pos['y']:.2f}), "
                    f"facing {ori['direction']}. Nearest: {nearest}, ~{dist:.2f} m away."
                    if ori else
                    f"I'm at ({pos['x']:.2f}, {pos['y']:.2f}). "
                    f"Nearest: {nearest}, ~{dist:.2f} m away."
                )
        else:
            response = "I haven't received position data yet."
        self.output_response(response)
        self.add_to_history(user_input, response, "position_query")

    def handle_location_query(self, user_input: str):
        lower = user_input.lower().strip()
        keywords = ['where is', 'where is the', "where's", "where's the",
                    'location of', 'find', 'show me where',
                    'coordinates of', 'position of', 'where can i find']
        location_name = None
        for kw in keywords:
            if kw in lower:
                location_name = lower.split(kw, 1)[1].strip().rstrip('?').strip()
                break
        if not location_name:
            response = "Which location are you asking about? Try: 'where is [location]?'"
            self.output_response(response)
            self.add_to_history(user_input, response, "location_query_unclear")
            return

        if self.is_partial_location_term(location_name):
            self.handle_partial_location_search(location_name, user_input)
            return

        matches = self.location_extractor.find_matching_locations(location_name)

        if not matches:
            _LOC_WORDS = frozenset(['office','lab','gate','room','store','bank',
                                    'canteen','warehouse','lawn','hall','block',
                                    'wing','dept','department'])
            _STOP = frozenset(['the','a','an','of','and','or','in','at','to','for','is','are'])
            _loc_toks = {tok for loc in self.location_manager.get_all_locations()
                         for tok in loc.lower().split()
                         if len(tok) >= 2 and tok not in _STOP}
            subject_words = set(location_name.lower().split())
            is_loc = bool((subject_words & _LOC_WORDS) or (subject_words & _loc_toks))
            if is_loc:
                response = self.response_generator.generate_response(user_input, status="unknown")
            else:
                response = self.response_generator.generate_response(
                    user_input, conversation_history=self.conversation_history)
            self.output_response(response)
            self.add_to_history(user_input, response, "chat")

        elif len(matches) == 1:
            d = self.location_manager.get_location(matches[0])
            response = f"{matches[0]} is at ({d['x']:.2f}, {d['y']:.2f})."
            pos = self.robot_navigator.get_current_position()
            if pos:
                dist, _ = self.calculate_distance_and_steps(pos, d)
                response += f" About {dist:.2f} m from your current position."
            self.output_response(response)
            self.add_to_history(user_input, response, "location_query")

        else:
            selected = self.location_extractor.disambiguate_location(matches)
            if selected:
                d = self.location_manager.get_location(selected)
                response = f"{selected} is at ({d['x']:.2f}, {d['y']:.2f})."
                pos = self.robot_navigator.get_current_position()
                if pos:
                    dist, _ = self.calculate_distance_and_steps(pos, d)
                    response += f" About {dist:.2f} m away."
                self.output_response(response)
                self.add_to_history(user_input, response, "location_query")

    def is_partial_location_term(self, term: str) -> bool:
        return term.lower() in ('office','offices','lab','labs','laboratory','laboratories')

    # ═══════════════════════════════════════════════════════════════
    #  Mode switch / stop / simple responses
    # ═══════════════════════════════════════════════════════════════

    def check_for_stop_command(self, user_input: str) -> bool:
        stop_cmds = ['stop','stop talking','be quiet','quiet','shut up',
                     'stop speaking','silence','enough','cancel','stop it']
        return any(cmd in user_input.lower().strip() for cmd in stop_cmds)

    def handle_mode_switch_command(self, user_input: str) -> bool:
        lower = user_input.lower().strip()
        voice_on  = ['switch to voice','voice mode','enable voice','use voice',
                     'talk to me','speak to me','voice on']
        voice_off = ['switch to text','text mode','disable voice','use text',
                     'type mode','text only','voice off']
        if any(t in lower for t in voice_on):
            if not (self.voice_input and self.voice_output):
                self.output_response("Voice mode is not available.")
            elif self.mode == 'voice':
                self.output_response("Already in voice mode!")
            else:
                self.set_mode('voice')
                self.output_response("Switched to voice mode.")
            return True
        if any(t in lower for t in voice_off):
            if self.mode == 'text':
                self.output_response("Already in text mode!")
            else:
                self.set_mode('text')
                self.output_response("Switched to text mode.")
            return True
        return False

    def handle_simple_response(self, user_input: str) -> str | None:
        lower = user_input.lower().strip()
        for category, triggers in {
            'yes':   ['yes','yeah','yep','yup','sure','ok','okay','alright'],
            'no':    ['no','nope','nah','not really'],
            'thanks':['thanks','thank you'],
        }.items():
            if lower in triggers:
                if self.last_response_asked_question:
                    return None
                return {"yes":   "Great! What else can I help you with?",
                        "no":    "Understood. Let me know if you need anything!",
                        "thanks":"You're welcome! Happy to help."}[category]
        return None

    # ═══════════════════════════════════════════════════════════════
    #  Voice cancel listener (terminal voice mode only)
    # ═══════════════════════════════════════════════════════════════

    def _voice_cancel_navigation_listener(self):
        if self.gui_mode:
            while not self.navigation_cancel_requested:
                time.sleep(0.2)
            return
        import sounddevice as sd_cn
        import numpy as np_cn
        CANCEL_WORDS = ["stop", "cancel", "abort", "halt"]
        native_sr = getattr(self.voice_input, '_native_sr', 48000)
        dev       = getattr(self.voice_input, '_preferred_input_device', None)
        try:
            while not self.navigation_cancel_requested:
                kw = dict(samplerate=native_sr, channels=1, dtype='float32')
                if dev is not None:
                    kw['device'] = dev
                chunk = sd_cn.rec(int(1.5 * native_sr), **kw)
                sd_cn.wait()
                flat = chunk.flatten()
                if np_cn.max(np_cn.abs(flat)) < 0.02:
                    continue
                if self.voice_input:
                    audio_16k = self.voice_input._resample_if_needed(flat)
                    text = self._transcribe_flat(audio_16k)
                    if text and any(cw in text.lower() for cw in CANCEL_WORDS):
                        self.logger.info(f"Cancel word: '{text}'")
                        self.cancel_navigation()
                        print("\n⛔ Navigation cancelled by voice!\n")
                        return
        except Exception as e:
            self.logger.error(f"Cancel listener error: {e}")

    def _transcribe_flat(self, audio_16k) -> str:
        if not self.voice_input or not self.voice_input._model:
            return ""
        try:
            if getattr(self.voice_input, '_USE_FASTER', False):
                segs, _ = self.voice_input._model.transcribe(audio_16k, language="en")
                return " ".join(s.text for s in segs).strip()
            return self.voice_input._model.transcribe(audio_16k, language="en").get("text","").strip()
        except Exception:
            return ""

    # ═══════════════════════════════════════════════════════════════
    #  Main entry point
    # ═══════════════════════════════════════════════════════════════

    def run(self):
        self.logger.info("Chat interface started.")
        self.chat_loop()

    def chat_loop(self):
        print(f"Current mode: {self.mode.upper()}\n")

        # Available-location substring triggers
        _LOC_LIST_TRIGGERS = (
            'available location','list location','show location',
            'what location','all location','which location',
            'where can you go','where can i go',
            'location on the map','location in the map',
            'tell me location','show me location',
            'tell me the location','what are the location',
            'what places','all places','list places','show places',
        )

        while True:
            try:
                # ── Check pending arrival announcements ────────────────
                self._process_arrivals()

                user_input = self.get_user_input()
                if not user_input:
                    continue

                lower_input = user_input.lower().strip()

                # ── Exit ───────────────────────────────────────────────
                if lower_input in ('quit','exit','bye','goodbye'):
                    self.output_response("Goodbye! Have a great day! 👋")
                    break

                # ── Stop speech ────────────────────────────────────────
                if self.check_for_stop_command(user_input):
                    if self.voice_output and self.voice_output.is_speaking():
                        self.voice_output.request_stop()
                        print("TURTLEBOT: Stopped.\n")
                        self.add_to_history(user_input, "Stopped.", "stop_command")
                    else:
                        self.output_response("I'm not speaking right now.")
                    continue

                # ── Mode switch ────────────────────────────────────────
                if self.handle_mode_switch_command(user_input):
                    continue

                # ── Location list ──────────────────────────────────────
                if (lower_input == 'locations'
                        or any(t in lower_input for t in _LOC_LIST_TRIGGERS)):
                    self.location_manager.show_all_locations()
                    names = self.location_manager.get_all_locations()
                    spoken = (", ".join(names[:-1]) + f", and {names[-1]}"
                              if len(names) > 1 else names[0])
                    response = f"I can navigate to {len(names)} locations: {spoken}."
                    self.output_response(response)
                    self.add_to_history(user_input, response, "show_locations")
                    continue

                # ── Go back ────────────────────────────────────────────
                if self.handle_go_back_command(user_input):
                    continue

                # ── Pending confirmation ───────────────────────────────
                if self.pending_navigation:
                    if self.handle_confirmation_response(user_input):
                        continue

                # ── Pending disambiguation ─────────────────────────────
                if self.pending_disambiguation:
                    matches = self.pending_disambiguation['matches']
                    orig    = self.pending_disambiguation['original_input']
                    result  = self._process_disambiguation_selection(
                        user_input, matches, orig)
                    if result == "done":
                        self.pending_disambiguation = None
                    continue

                # ── Guard bare numbers ─────────────────────────────────
                if user_input.strip().rstrip('.').isdigit():
                    self.output_response(
                        "That looks like a number selection. "
                        "Please use this only when I show you options to choose from.")
                    continue

                # ── Detect intent ──────────────────────────────────────
                intent = self.intent_detector.detect_intent(user_input)

                # Partial location term (lab / office)
                if self.is_partial_location_term(lower_input):
                    self.handle_partial_location_search(lower_input, user_input)
                    continue

                # ── UNCERTAIN intent ───────────────────────────────────
                if intent == 'uncertain':
                    if self.is_partial_location_term(user_input):
                        self.handle_partial_location_search(user_input, user_input)
                        continue

                    matches = self.location_extractor.find_matching_locations(user_input)
                    if matches:
                        result = self.confirm_partial_navigation(matches)
                        if result == "WAITING_CONFIRMATION":
                            continue
                        continue

                    _LOC_WORDS = frozenset([
                        'office','lab','gate','room','store','bank',
                        'canteen','warehouse','lawn','lounge','hall',
                        'block','wing','floor','building','dept','department',
                    ])
                    _STOP = frozenset(['the','a','an','of','and','or','in','at','to','for','is','are','my','me'])
                    _loc_toks = {tok for loc in self.location_manager.get_all_locations()
                                 for tok in loc.lower().split()
                                 if len(tok) >= 2 and tok not in _STOP}
                    if (set(lower_input.split()) & _LOC_WORDS) or \
                       (set(lower_input.split()) & _loc_toks):
                        response = self.response_generator.generate_response(
                            user_input, status="unknown")
                        self.output_response(response)
                        self.add_to_history(user_input, response, "location_not_found")
                        continue

                    intent = 'chat'

                # ── CHAT intent ────────────────────────────────────────
                if intent == 'chat':
                    loc_query_patterns = [
                        'where is',"where's",'location of','coordinates of',
                        'where can i find','show me where','position of',
                    ]
                    if any(p in lower_input for p in loc_query_patterns):
                        _subj = lower_input
                        for _p in loc_query_patterns:
                            if _p in _subj:
                                _subj = _subj.split(_p, 1)[1].strip().rstrip('?').strip()
                                break
                        if (self.location_extractor.find_matching_locations(_subj)
                                or self.is_partial_location_term(_subj)):
                            self.handle_location_query(user_input)
                        else:
                            response = self.response_generator.generate_response(
                                user_input, conversation_history=self.conversation_history)
                            self.output_response(response)
                            self.add_to_history(user_input, response, "chat")
                        continue

                    pos_queries = [
                        'where am i', 'where are you', 'where are we',
                        'current position', 'current location',
                        'my position', 'your position', 'our position',
                        'where is the robot', 'where is robot', 'robot location',
                        'robot position', 'what is your location', 'what is my location',
                        'tell me your position', 'tell me my position',
                        'show my position', 'show your position',
                        'what are my coordinates', 'what are your coordinates',
                        'where are you now', 'where am i now', 'where are you at',
                        'your current location', 'my current location',
                        'give me your location', 'give me my location',
                    ]
                    if lower_input in pos_queries:
                        self.handle_position_query(user_input)
                        continue

                    for greeting, resp in [
                        (['hi','hello','hey','hi there','hello there'],
                         "Hello! How can I assist you today?"),
                        (['how are you','how are you doing',"how's it going"],
                         "I'm doing great, thank you! How can I help you?"),
                        (['thanks','thank you','thanks a lot','thank you very much'],
                         "You're welcome! Let me know if you need anything else."),
                    ]:
                        if lower_input in greeting:
                            self.output_response(resp)
                            self.add_to_history(user_input, resp, "greeting")
                            break
                    else:
                        simple = self.handle_simple_response(user_input)
                        if simple:
                            self.output_response(simple)
                            self.add_to_history(user_input, simple, "acknowledgment")
                        else:
                            # Tag personal/interpersonal queries so LLM
                            # gives empathetic rather than factual answers
                            _personal_kws = [
                                'my friend', 'my boss', 'my colleague',
                                'my coworker', 'my manager', 'my supervisor',
                                'friend is', 'boss is', 'colleague is',
                            ]
                            tagged = user_input
                            if any(kw in lower_input for kw in _personal_kws):
                                tagged = f"[PERSONAL_PROBLEM] {user_input}"
                            response = self.response_generator.generate_response(
                                tagged, conversation_history=self.conversation_history)
                            self.output_response(response)
                            self.add_to_history(user_input, response, "chat")
                    continue

                # ── NAVIGATE intent ────────────────────────────────────
                # Try multi-location first ("go to home then lawn")
                multi = self.parse_multi_location(user_input)
                if multi:
                    self._queue_navigation(multi, user_input)
                    continue

                # Single location
                location = self.location_extractor.extract_location(user_input)
                if location:
                    if isinstance(location, dict) and 'is_coordinate' in location:
                        dest_str = f"coordinates ({location['x']}, {location['y']})"
                    else:
                        dest_str = location

                    if self._is_already_at_target(location):
                        source_loc = self.show_navigation_context(location)
                        response = f"I'm already at {dest_str}!"
                        self.output_response(response)
                        self.add_to_history(user_input, response,
                                            "already_there", location, source_loc)
                    else:
                        self._queue_navigation([location], user_input)
                else:
                    response = ("Location not found. Please specify valid coordinates "
                                "(x: 0–8, y: 0–3) or choose from available locations.")
                    self.output_response(response)
                    self.location_manager.show_all_locations()
                    self.add_to_history(user_input, response, "location_not_found")

            except KeyboardInterrupt:
                print("\n\nTURTLEBOT: Shutting down... 👋")
                break
            except Exception as e:
                self.logger.error(f"Chat loop error: {e}")
                self.output_response(f"Sorry, I encountered an error: {e}")