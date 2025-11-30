# `game.py` – Core Engine Structures and What Agent Authors Need

This file defines the **core engine** types used by the Capture the Flag project.  
As an agent author, the most relevant parts are:

- `Agent` (base agent interface)
- `Directions` (standard move labels)
- `Configuration` (position + direction of an agent)
- `AgentState` (state of a single agent: pos, pacman/ghost, scared, carrying food)
- `Grid` (for walls and food grids)
- `Actions` (movement helpers and legal-move logic)
- `GameStateData` (internal representation of state)
- `Game` (main game loop that calls your agent)

You usually won’t modify this file, but understanding these types helps you reason about the environment and how your agent is called.

---

## 1. `Agent` – Base Class for All Agents

Class: `Agent`

- Constructor:
  - `__init__(self, index=0)`  
    - Stores `self.index` (your agent’s index).
- Method to implement:
  - `get_action(self, state)`  
    - Must return one of the actions: `Directions.NORTH`, `SOUTH`, `EAST`, `WEST`, `STOP`.

In Capture the Flag, you inherit from `CaptureAgent` (which itself inherits from `Agent`), but `Agent` defines the core interface.

---

## 2. `Directions` – Canonical Move Labels

Class: `Directions`

- Constants:
  - `NORTH`, `SOUTH`, `EAST`, `WEST`, `STOP` – strings used to represent actions.

- Helpers:
  - `LEFT`: mapping of direction → left turn.
  - `RIGHT`: mapping of direction → right turn.
  - `REVERSE`: mapping of direction → reverse direction.

You use these to compare or choose moves, e.g. avoid `REVERSE[my_direction]` to discourage going backwards.

---

## 3. `Configuration` – Position + Direction of an Agent

Class: `Configuration`

Represents the **position and direction** of an agent at a specific moment.

- Constructor:
  - `__init__(self, pos, direction)`
    - `pos`: `(x, y)` coordinates (can be float, not always integers mid-move).
    - `direction`: one of `Directions.*`.

- Methods:
  - `get_position()`: returns `pos`.
  - `get_direction()`: returns `direction`.
  - `is_integer()`: returns `True` if `(x, y)` are exact integers (agent on a grid point).
  - `generate_successor(vector)`:
    - Returns a new `Configuration` by moving along a given `(dx, dy)` vector.
    - Uses `Actions.vector_to_direction(vector)` to infer the new direction.
    - If the direction is `STOP`, keeps the old direction.

Usage idea for agent authors:
- You don’t usually create `Configuration` yourself, but you interact with it via:
  - `AgentState.get_position()`
  - `AgentState.get_direction()`
- Understanding that positions can be fractional explains why `CaptureAgent.get_action` checks `nearest_point` before calling `choose_action`.

---

## 4. `AgentState` – Per-Agent Dynamic State

Class: `AgentState`

Represents the **dynamic state** of a single agent (Pacman or ghost):

- Constructor:
  - `__init__(self, start_configuration, is_pacman)`
    - `start`: starting `Configuration`.
    - `configuration`: current `Configuration` (can be `None` if hidden).
    - `is_pacman`: boolean (Pacman vs ghost).
    - `scared_timer`: integer, how long the agent remains scared.
    - `num_carrying`: how many food pellets currently being carried (as Pacman).
    - `num_returned`: how many food pellets have been successfully brought home.

- Methods:
  - `copy()`: returns a deep copy of this `AgentState`.
  - `get_position()`: returns `(x, y)` if `configuration` exists, otherwise `None`.
  - `get_direction()`: returns direction from `configuration`.

As an agent author, you most often get `AgentState` via:

- `game_state.get_agent_state(index)`

Fields you might inspect:

- `is_pacman`: are you currently Pacman or ghost?
- `scared_timer`: are you scared (and for how long)?
- `num_carrying`: how much food you’re carrying.
- `num_returned`: total returned so far by that agent.

---

## 5. `Grid` – 2D Boolean Matrix for Walls and Food

Class: `Grid`

Represents a 2D array of booleans, used primarily for:
- Food (where `True` indicates presence of food).
- Walls (where `True` indicates a wall).

Key properties and methods:

- Constructor:
  - `__init__(self, width, height, initial_value=False, bit_representation=None)`
    - Creates a `width x height` grid initialized with `initial_value`.
    - Can also be reconstructed from a compressed bit representation.

- Indexing:
  - `grid[x][y]` with `0 <= x < width`, `0 <= y < height`.

- Important methods:
  - `copy()`, `deep_copy()`, `shallow_copy()`.
  - `count(item=True)`: counts how many cells equal `item`.
  - `as_list(key=True)`: returns a list of `(x, y)` where `grid[x][y] == key`.

As an agent author, you mainly encounter `Grid` via:

- `game_state.get_walls()`
- `game_state.get_red_food()`, `game_state.get_blue_food()`

Typical usage:
- Iterate over all positions and check `walls[x][y]`, `food[x][y]`.
- Use `food.as_list(True)` to get all food positions.

---

## 6. `Actions` – Movement and Legal Moves

Class: `Actions`

Provides static methods to work with directions, vectors, and legal moves.

Important attributes:

- `_directions`:
  - Mapping from direction string → `(dx, dy)` vector.
- `TOLERANCE`:
  - Small numeric tolerance used to decide if an agent is exactly on a grid point.

Important methods:

- `reverse_direction(action)`:
  - Returns the reverse of a given direction.

- `vector_to_direction(vector)`:
  - Maps `(dx, dy)` to a direction string or `STOP`.

- `direction_to_vector(direction, speed=1.0)`:
  - Returns `(dx, dy)` for given direction and speed.

- `get_possible_actions(config, walls)`:
  - Given a `Configuration` and `walls` `Grid`, returns a list of possible directions.
  - Logic:
    - If the agent is **between** grid points (`abs(x - round(x)) + abs(y - round(y)) > TOLERANCE`):
      - Only allowed to continue in its current direction.
    - If exactly on a grid point:
      - For each direction, checks if the neighboring cell has no wall.
      - Adds those directions to the possible list.

- `get_legal_neighbors(position, walls)`:
  - Returns a list of neighboring positions (grid points) reachable from `position` without hitting walls.

- `get_successor(position, action)`:
  - Returns the new `(x, y)` after taking `action` from `position`.

These methods underlie:

- `GameState.get_legal_actions(agent_index)` in `capture.py`, because that calls `Actions.get_possible_actions`.
- Pathfinding logic if you want to manually explore neighbors instead of using `get_maze_distance`.

---

## 7. `GameStateData` – Internal State Container

Class: `GameStateData`

Holds the **internal representation** of the game state. You usually don’t interact with it directly, but `GameState.data` is an instance of this.

Key fields:

- `food`: `Grid` of food.
- `capsules`: list of `(x, y)` capsule positions.
- `agent_states`: list of `AgentState` objects (one per agent).
- `layout`: the layout object (contains walls, starting positions, etc).
- `score`: current score.
- `score_change`: score delta at each step.
- `timeleft`: moves remaining.
- `_food_eaten`, `_food_added`, `_capsule_eaten`: markers for UI and logic.
- `_win`, `_lose`: flags for terminal states.
- `_agent_moved`: index of the agent that moved last.

Important methods:

- Constructor:
  - Copies fields from previous state if provided (for `deep_copy` behavior).
- `deep_copy()`:
  - Creates a deep copy, copying food, layout, agent states, etc.

- `copy_agent_states(agent_states)`:
  - Helper to deep copy the agent states.

- `initialize(layout, num_ghost_agents)`:
  - Creates the initial state given a layout and number of ghost agents:
    - Copies food and capsules from `layout`.
    - Sets score to 0.
    - Creates `AgentState`s at starting positions from `layout.agent_positions`.

- `__str__()`:
  - Builds a text representation of the board with:
    - `%` = wall
    - `.` = food
    - `G` = ghost
    - `v`, `^`, `<`, `>` = Pacman (directional).
    - `o` = capsule

This is what backs `str(game_state.data)` and the text-based visualization.

For agent authors, it’s mainly useful accidentally (debug printouts) and to understand what’s happening inside `GameState`, but you work directly with `GameState` methods instead of accessing `GameStateData` manually.

---

## 8. `Game` – Main Game Loop that Calls Your Agent

Class: `Game`

Manages the **flow of the game**:
- Calling agents for actions.
- Applying those actions to the state.
- Handling timeouts and crashes.
- Updating the display.
- Applying game rules (win/loss).

Constructor:

- `__init__(self, agents, display, rules, starting_index=0, mute_agents=False, catch_exceptions=False)`

Important fields:

- `self.agents`: list of agent instances (yours + opponents).
- `self.display`: a display object for rendering (text or graphics).
- `self.rules`: rule controller (for Capture, this is `CaptureRules` in `capture.py`).
- `self.state`: current `GameState` (set externally).
- Timing and logging:
  - `self.total_agent_times`
  - `self.total_agent_time_warnings`
  - `self.agent_timeout`
  - `self.move_history`: list of `(agent_index, action)` pairs.

Key methods:

### 8.1. `run(self, delay=0)`

This is the main game loop.

High-level flow:

1. `self.display.initialize(self.state.data)`
2. Calls `register_initial_state` on each agent (if defined):

   - `agent.register_initial_state(self.state.deep_copy())`
   - Timed with a startup time limit from `rules.get_max_startup_time()`.

3. Sets `agent_index = starting_index`.

4. While `not self.game_over`:
   - Sleeps for `delay` seconds (if given).
   - Picks the current `agent = self.agents[agent_index]`.

   - Observation step:
     - If the agent defines `observation_function`:
       - Calls `agent.observation_function(self.state.deep_copy())` (with a move timeout).
     - Otherwise:
       - Uses the full `self.state.deep_copy()` as the observation.

   - Action step:
     - Calls `agent.get_action(observation)` with time limits (`get_move_timeout`).
     - If the agent times out too often or exceeds total time:
       - `agent_timeout` is set and `rules.agent_crash` is invoked.

   - Apply action:
     - Records `(agent_index, action)` in `move_history`.
     - Updates state:
       - `self.state = self.state.generate_successor(agent_index, action)`
         (this calls the appropriate environment rules, e.g. `CaptureRules`).

   - Update display:
     - `self.display.update(self.state.data)`

   - Check game rules:
     - `self.rules.process(self.state, self)` (may set `self.game_over`).

   - Advance `agent_index`:
     - `(agent_index + 1) % num_agents`

   - If BOINC is enabled, updates progress.

5. After exiting the loop:
   - Calls `agent.final(self.state)` on each agent that defines `final`.

This explains:

- **When** your `register_initial_state` is called (once at game start).
- **When** your `observation_function` and `get_action` are called (once per turn).
- **Why** you must respect time limits (or your agent will crash/forfeit).
- **How** `generate_successor` is used: the engine applies your action to produce the next state.

### 8.2. Time and crash handling

The game uses `TimeoutFunction` to enforce time limits, based on `rules`:

- `get_max_startup_time()` – total time allowed for `register_initial_state`.
- `get_move_timeout()` – hard timeout for each move.
- `get_move_warning_time()` – soft warning threshold per move.
- `get_max_total_time()` – maximum accumulated time allowed.

If your agent exceeds any critical limit, `Game` will:

- Print warnings or errors.
- Mark `agent_timeout = True`.
- Call `rules.agent_crash(self, agent_index)`.

Understanding this helps you design agents that are efficient and do not risk disqualification due to timeouts.

---

## 9. Summary for Agent Authors

From `game.py`, the key takeaways are:

- `Agent` defines the base `get_action` signature. You indirectly inherit from this via `CaptureAgent`.
- `Directions` are the canonical action names you should always return.
- `Configuration` and `AgentState` define how agent positions, directions, and pacman/ghost/scared/carrying status are represented.
- `Grid` is the 2D boolean container used for food and walls; you inspect it through `GameState` methods.
- `Actions` defines how directions relate to vectors and which moves are legal based on walls and between-tile positioning.
- `GameStateData` is the internal state container behind `GameState`, mainly useful for debugging and understanding how state is stored.
- `Game` manages the game loop, timeouts, and calls to:
  - `register_initial_state`
  - `observation_function`
  - `get_action`
  - `final`

Combined with:

- `capture.py` (Capture rules + `GameState` API)
- `capture_agents.py` (`CaptureAgent` interface and helpers)

you have all the conceptual pieces you need to understand **how your agents are run**, what information they receive, and how their actions transform the game state.

