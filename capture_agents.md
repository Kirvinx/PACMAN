# `capture_agents.py` – Interfaces and Base Classes for Capture-the-Flag Agents

This file defines the **interfaces and base classes** that you will subclass to implement your own capture-the-flag agents.

Authors of custom agents mainly need to understand:

- `CaptureAgent` (core base class you subclass)
- Its lifecycle: `__init__`, `register_initial_state`, `get_action`, `choose_action`
- Utility methods (food, capsules, team/opponents, score, maze distance, observations)
- Optional visualization utilities (for debugging beliefs)
- Simpler baseline agents: `RandomAgent`, `TimeoutAgent`
- Legacy `AgentFactory` (mostly irrelevant for new code)

---

## 1. Imports and Dependencies

Key imports:

- `random`: for random choice in baseline agents.
- `contest.distance_calculator as distance_calculator`: provides the `Distancer` class for maze distances.
- `contest.util as util`: general utilities, including `Counter`, `raise_not_defined`.
- `contest.game.Agent`: base agent class from the game engine.
- `contest.util.nearest_point`: snaps positions to the nearest grid point.

You don’t usually need to modify these imports; you just use the provided functionality in your own agents.

---

## 2. `AgentFactory` (Legacy / Backwards Compatibility)

Class:

- `AgentFactory`:
  - Constructor: `__init__(self, is_red, **args)`
    - Stores `self.is_red` (whether the factory is for the red team).
  - Method: `get_agent(self, index)`
    - Raises `util.raise_not_defined()`.
  - This class is not used in the current setup but kept for backward compatibility with older team submissions that imported it.

You do not need this for normal agent development.

---

## 3. `RandomAgent`: A Minimal Example Agent

Class:

- `RandomAgent(Agent)`:
  - Constructor: `__init__(self, index)`
    - Calls `super().__init__(index)`.
    - Stores `self.index`.
  - Method: `get_action(self, state)`
    - Returns a random legal action using:
      - `state.get_legal_actions(self.index)`
      - `random.choice(...)`

This is a simple example of an agent that:
- Ignores strategy.
- Only uses the standard `Agent` interface and `GameState`.
- Does not use any of the Capture-specific helpers.

It’s useful as a reference for the simplest possible agent.

---

## 4. `CaptureAgent`: The Main Base Class You Should Subclass

`CaptureAgent` is the key class for writing capture-the-flag agents. You will almost always subclass this one.

Purpose:

- Provides:
  - Team info (red/blue, teammates, opponents).
  - Convenience access to food/capsules.
  - A cached maze distance calculator (`Distancer`).
  - Observation history (for tracking).
  - A standard action cycle (`get_action` → `choose_action`).
  - Optional visualization of probability distributions.

Recommended usage:

- Subclass `CaptureAgent`
- Override `choose_action(self, game_state)` with your own logic.

### 4.1. Initialization

Constructor:

- `__init__(self, index, time_for_computing=.1)`

Sets up the following attributes:

- `self.index`:
  - The agent’s index (0–3 usually in 2v2).
- `self.red`:
  - `True` if this agent is on the red team, `False` otherwise.
  - Initially `None`, set later in `register_initial_state`.
- `self.agentsOnTeam`:
  - A list of indices for your team’s agents.
  - Set via `register_team`.
- `self.distancer`:
  - Will store a `Distancer` instance that can compute maze distances.
- `self.observation_history`:
  - A list of `GameState` objects representing observed states over time.
- `self.timeForComputing`:
  - A time budget (in seconds) for computing maze distances when initializing `distancer`.
- `self.display`:
  - Handle to the display, if available, for drawing debug info.
- `self._distributions`:
  - For storing distributions when display-based visualization is not available.

You don’t normally override `__init__`; you use these fields in your agent logic.

---

### 4.2. `register_initial_state(self, game_state)`

Called once at the beginning of a game for initial setup.

Actions:

1. Sets `self.red` using:
   - `game_state.is_on_red_team(self.index)`
2. Calls `self.register_team(self.get_team(game_state))` to store your team’s agent indices.
3. Creates a distance calculator:
   - `self.distancer = distance_calculator.Distancer(game_state.data.layout)`
4. Calls:
   - `self.distancer.get_maze_distances()`
   - This precomputes all-pairs maze distances.
   - If you comment this out, only Manhattan distance will be used.
5. Sets up graphics access:
   - Imports `__main__` and, if `_display` is in its namespace, stores it in `self.display`.

You can override this method to add custom initialization, but usually you call `super().register_initial_state(game_state)` first to keep this setup.

---

### 4.3. `final(self, game_state)`

Called at the end of a game.

- Clears:
  - `self.observation_history = []`

You can override this to add end-of-game behavior, but this is optional.

---

### 4.4. `register_team(self, agents_on_team)`

- Sets:
  - `self.agentsOnTeam = agents_on_team`

This method is called from `register_initial_state` to store your team’s indices.

---

### 4.5. Observation Hook: `observation_function`

Method:

- `observation_function(self, game_state)`

Returns:

- `game_state.make_observation(self.index)`

Meaning:

- This defines how the agent observes the game state:
  - It uses the `GameState`’s partial observability model (noisy distances, hidden distant enemies).
- Changing this would affect what your agent sees when running under `capture.py`.

You normally leave this as is, so the standard observation logic applies.

---

### 4.6. Debug Draw Methods

Useful for visual debugging when using graphical display.

- `debug_draw(self, cells, color, clear=False)`:
  - If `self.display` is set and is an instance of `PacmanGraphics`:
    - Ensures `cells` is a list.
    - Calls `self.display.debug_draw(cells, color, clear)`.
  - Used to draw markers (e.g., suspected enemy positions) on the board.

- `debug_clear(self)`:
  - Clears any debug drawings using `self.display.clear_debug()`.

These do not affect gameplay; they’re purely visual helpers.

---

### 4.7. Action Selection: `get_action` and `choose_action`

This is the core of how your agent acts during the game.

#### `get_action(self, game_state)`

You normally do NOT override this.

Behavior:

1. Appends the current `game_state` to `self.observation_history`.
2. Retrieves your agent state and position:
   - `my_state = game_state.get_agent_state(self.index)`
   - `my_pos = my_state.get_position()`
3. Checks if you are exactly at a grid point:
   - Uses `nearest_point(my_pos)`
   - If `my_pos != nearest_point(my_pos)`:
     - You are “between” grid points (on a half-step).
     - Returns the first legal action:
       - `game_state.get_legal_actions(self.index)[0]`
       - This continues your previous direction, typically.
   - Else:
     - Calls `self.choose_action(game_state)` and returns its result.

Meaning:

- Your custom `choose_action` is only called when the agent is exactly on a discrete tile, not mid-movement.
- `get_action` also handles bookkeeping of observation history.

#### `choose_action(self, game_state)`

Intended for you to override.

- Default implementation:
  - `util.raise_not_defined()`
- You implement your own decision logic here and return a legal action.

Example typical pattern in subclasses:

- Get legal actions:
  - `actions = game_state.get_legal_actions(self.index)`
- Evaluate each action (using features, search, etc.).
- Return the best action according to your evaluation.

---

## 5. Convenience Methods for Game Data

These helpers make it easier to write your agent logic.

### 5.1. Food Access

- `get_food(self, game_state)`:
  - Returns the food you are supposed to eat.
  - If you are red, this is blue’s food:
    - `game_state.get_blue_food()`
  - If you are blue, this is red’s food:
    - `game_state.get_red_food()`
  - Returns a grid `m` where `m[x][y]` is true if there is enemy food to eat at `(x, y)`.

- `get_food_you_are_defending(self, game_state)`:
  - Returns the food on your side that you must defend.
  - If you are red:
    - `game_state.get_red_food()`
  - If you are blue:
    - `game_state.get_blue_food()`

These methods automatically flip based on the team so you don’t have to.

---

### 5.2. Capsule Access

- `get_capsules(self, game_state)`:
  - Capsules you can eat (on the enemy side).
  - If you are red:
    - `game_state.get_blue_capsules()`
  - If you are blue:
    - `game_state.get_red_capsules()`

- `get_capsules_you_are_defending(self, game_state)`:
  - Capsules on your side which you must defend.
  - If you are red:
    - `game_state.get_red_capsules()`
  - If you are blue:
    - `game_state.get_blue_capsules()`

---

### 5.3. Teams and Opponents

- `get_opponents(self, game_state)`:
  - Returns a list of agent indices for your opponents.
  - If you are red:
    - `game_state.get_blue_team_indices()`
  - If you are blue:
    - `game_state.get_red_team_indices()`

- `get_team(self, game_state)`:
  - Returns a list of agent indices for your team.
  - If you are red:
    - `game_state.get_red_team_indices()`
  - If you are blue:
    - `game_state.get_blue_team_indices()`

These help you quickly figure out who is friend vs foe.

---

### 5.4. Score Perspective

- `get_score(self, game_state)`:
  - Returns how much your team is beating the other team by.
  - If you are red:
    - Returns `game_state.get_score()`.
  - If you are blue:
    - Returns `game_state.get_score() * -1`.

So:
- Positive value = your team is winning.
- Negative value = your team is losing.
- Zero = tie.

This avoids you having to remember which sign corresponds to which team.

---

### 5.5. Maze Distance

- `get_maze_distance(self, pos1, pos2)`:
  - Uses `self.distancer` to compute the distance between two positions.
  - If `get_maze_distances()` was called in `register_initial_state`, this returns:
    - True shortest-path distance (maze distance) accounting for walls.
  - Otherwise:
    - Falls back to Manhattan distance.

Usage:
- Prefer `get_maze_distance` over Manhattan distance when pathing or evaluating distances, especially for navigation-heavy behavior.

---

### 5.6. Observations History

- `get_previous_observation(self)`:
  - Returns the previous `GameState` your agent observed (the one before the current move).
  - If the history length is 1 (i.e., first move), returns `None`.

- `get_current_observation(self)`:
  - Returns the most recent observation:
    - `self.observation_history[-1]`

Note:
- Observations are partial game states produced by `observation_function` / `make_observation`, so they might not include exact locations of all opponents.

These are helpful for tracking changes over time (e.g., “was there food here last turn?” or “did an opponent just disappear from vision?”).

---

### 5.7. Visualization of Belief Distributions

- `display_distributions_over_positions(self, distributions)`:

Purpose:
- Visual debugging of your belief about agent locations.

Input:
- `distributions` is a list or tuple where each element is either:
  - A `util.Counter` mapping `(x, y)` positions to probabilities for a given agent.
  - Or `None` if you don’t want to display anything for that agent.

Behavior:
- Verifies each non-None element is a `util.Counter`.
- Builds a list of counters (replacing None with empty counters).
- If `self.display` has `update_distributions`:
  - Calls `self.display.update_distributions(dists)`.
- Else:
  - Stores `self._distributions = dists` (so external tools like `pacclient.py` can read them).

Effect:
- Overlays probability distribution heatmaps on the board.
- Very useful for debugging probabilistic tracking algorithms.

Does not affect gameplay.

---

## 6. `TimeoutAgent`: Example of a Slow Agent

Class:

- `TimeoutAgent(Agent)`:
  - Constructor: `__init__(self, index)`
    - Calls `super().__init__(index)`.
    - Stores `self.index`.
  - Method: `get_action(self, state)`:
    - Imports `time` and `random`.
    - Calls `time.sleep(2.0)` (waits 2 seconds).
    - Returns `random.choice(state.get_legal_actions(self.index))`.

Purpose:
- Demonstrates what happens when an agent takes too long to move.
- The game enforces timeouts and will penalize or randomize actions if your agent is too slow.
- You should not base your agent on this; it’s mainly a stress-test or example.

---

## 7. Summary for Agent Authors

If you are implementing your own team:

1. Subclass `CaptureAgent`, not `Agent`:
   - This gives you access to distances, team helpers, food/capsule helpers, observation history, and visualization.

2. Override:
   - `choose_action(self, game_state)`
   - Inside, use:
     - `get_food`, `get_food_you_are_defending`
     - `get_capsules`, `get_capsules_you_are_defending`
     - `get_opponents`, `get_team`
     - `get_score`
     - `get_maze_distance`
     - `get_current_observation`, `get_previous_observation`
     - Any custom inference or planning code you write.

3. Do not override `get_action` unless you know exactly what you’re doing:
   - It correctly handles half-step movement and observation history.

4. Use `register_initial_state` (optionally overriding it) to:
   - Set up anything that needs the layout or initial positions.
   - But keep the `distancer` setup unless you have a reason not to.

5. Use `display_distributions_over_positions` and `debug_draw`/`debug_clear` for:
   - Visual debugging of belief tracking or path planning.

Understanding this file gives you a solid foundation for writing sophisticated capture-the-flag agents while relying on a well-defined lifecycle and helper utilities.

