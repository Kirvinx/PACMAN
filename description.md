# `capture.py` – Key Concepts and Structure

This file defines the game logic, rules, and command-line interface for the Capture the Flag variant of Pacman. As an agent author, you mostly need to understand:

- The `GameState` interface (how your agent sees and simulates the world)
- How teams, food, capsules, and scoring work
- How partial observability and noisy distances work
- How deaths, scared timers, and food dropping behave
- How games are created and run (`CaptureRules`, `run_games`, `read_command`)
- How your team code is loaded (`load_agents`)

Everything else is infrastructure.

---

## 1. Global Constants and Utilities

Important constants:

- `KILL_POINTS`: points for killing a Pacman (set to `0` here).
- `SONAR_NOISE_RANGE`, `SONAR_NOISE_VALUES`: specify the noise added to distances.
- `SIGHT_RANGE`: Manhattan distance within which opponents are fully observable.
- `MIN_FOOD`: minimum food allowed on a side (used for win conditions).
- `TOTAL_FOOD`: total number of food pellets in the map.
- `DUMP_FOOD_ON_DEATH`: if `True`, when a Pacman dies, their carried food is dropped as new pellets.
- `SCARED_TIME`: number of moves an opponent is scared after capsule consumption.
- `COLLISION_TOLERANCE`: threshold distance for death collisions.

Utility function:

- `compute_noisy_distance(pos1, pos2)`: returns a noisy Manhattan distance between `pos1` and `pos2` based on `SONAR_NOISE_VALUES`.

Food / map partition helpers:

- `make_half_grid(grid, red)`: returns a grid with only the red or blue side’s food.
- `half_list(locations, grid, red)`: filters a list of positions to only those on red or blue side.

These control game physics and observations, and you may use them conceptually when designing your inference or evaluation.

---

## 2. `GameState`: The Main Interface for Agents

`GameState` is the core object your agents interact with. It wraps a `GameStateData` instance and provides methods to query and simulate the game.

### 2.1. Core methods you will use

These are the methods most relevant to agent code:

- `get_legal_actions(agent_index=0)`  
  Returns a list of legal moves for the given agent index. Internally calls `AgentRules.get_legal_actions`.

- `generate_successor(agent_index, action)`  
  Make a clone of the world, move the chosen Pacman one step in a direction, update everything that would change as a result, and give me that new world snapshot. Returns a new `GameState` resulting from the specified `agent_index` taking `action`. It:
  - Copies the current state.
  - Applies the action (`AgentRules.apply_action`).
  - Checks for deaths (`AgentRules.check_death`).
  - Decrements the scared timer of that agent (`AgentRules.decrement_timer`).
  - Updates score and remaining time.

  This is the function to use if you want to simulate future states.

- `get_agent_state(index)`  
  Returns the `AgentState` object for the given agent index (contains position, scared timer, pacman/ghost status, etc.). Scared timer is the amount of time that the pac is "invincible" (eats a power shell) and ghosts can't kill him. When it's 0, the pac isn't invincible.

- `get_agent_position(index)`  
  Returns `(x, y)` for that agent if observable, else `None` (due to partial observability).

- `get_num_agents()`  
  Returns total number of agents (always 4 in standard capture: 2 red, 2 blue).

- `get_score()`  
  Returns the current global score (positive for red advantage, negative for blue).

- `get_walls()`  
  Returns a `Grid` of walls.

- `has_food(x, y)` / `has_wall(x, y)`  
  Quick checks for food or wall at `(x, y)`.

- `get_capsules()`  
  Returns a list of positions of remaining capsules.

- `is_over()`  
  Indicates whether the game has ended.

### 2.2. Team and map partition methods

These methods help distinguish teams and map sides:

- `get_red_team_indices()` / `get_blue_team_indices()`  
  Returns a list of agent indices for each team.

- `is_on_red_team(agent_index)`  
  `True` if the agent belongs to the red team.

- `get_red_food()` / `get_blue_food()`  
  Returns grids representing food on each team’s side:
  - Red food: food red is protecting, blue wants to eat.
  - Blue food: food blue is protecting, red wants to eat.

- `get_red_capsules()` / `get_blue_capsules()`  
  Capsules on each side of the board.

- `get_initial_agent_position(agent_index)`  
  Starting position for that agent.

- `is_red(config_or_pos)`  
  Returns `True` if a given position or configuration is on the red side (x-coordinate less than half the width).

These are useful when designing offense/defense logic or detecting when you are Pacman/ghost.

### 2.3. Partial observability and noisy distances

The game is partially observable. These elements are crucial:

- `get_agent_distances()`  
  Returns a list of noisy distances (distances corrupted by some random noise) to each agent for the last observation, or `None` if not set. These are generated in `make_observation`.

- `@staticmethod get_distance_prob(true_distance, noisy_distance)`  
  Returns `P(noisy_distance | true_distance)` based on the noise model defined by `SONAR_NOISE_VALUES`. You should use this when building probabilistic models of opponents.

- `make_observation(index)`  
  Returns an observation for `agent index` as a new `GameState`:
  - Adds `agent_distances` (noisy distances from that agent to all others).
  - For enemies that are too far from any teammate (beyond `SIGHT_RANGE`), hides their positions by setting their configuration to `None`.
  - Keeps full information for your own team and enemies within `SIGHT_RANGE` of any teammate.

This method describes exactly what information your agent receives each turn.

### 2.4. Copying and initialization

- `__init__(prev_state=None)`  
  Creates a new `GameState`. If `prev_state` is provided, copies its data and team info.

- `deep_copy()`  
  Returns a deep copy of the current state, including food, agent states, teams, and time left.

- `initialize(layout, num_agents)`  
  Sets up an initial state from a `layout` and number of agents. Determines team membership (red vs blue) based on starting positions and sets `TOTAL_FOOD` according to the layout.

These are mainly internal, but understanding them helps you trust `generate_successor` and `make_observation` copies.

---

## 3. `CaptureRules`: Game-Level Flow and Termination

`CaptureRules` manages the whole game lifecycle (starting, ending, and time limits).

Key methods:

- `new_game(layout, agents, display, length, mute_agents, catch_exceptions)`  
  Creates and returns a `Game` object with:
  - An initial `GameState` from the given layout.
  - Randomly chosen starting team (red or blue).
  - Time limit (`length`) in moves (stored in `state.data.timeleft`).
  - Initial counts of each side’s food: `_init_blue_food`, `_init_red_food`.

- `process(state, game)`  
  Called after each move:
  - Checks if the game should end (by move count or win condition).
  - When the game ends (`state.is_over()`), prints a description of why:
    - Team reached required returned-food threshold: `(TOTAL_FOOD / 2) - MIN_FOOD`
    - Or time ran out, in which case high score wins or it is a tie.

- `agent_crash(game, agent_index)`  
  If an agent crashes, immediately sets score to `-1` (if red agent crashes) or `+1` (if blue agent crashes).

- Time bounds methods:
  - `get_max_total_time()`
  - `get_max_startup_time()`
  - `get_move_warning_time()`
  - `get_move_timeout()`
  - `get_max_time_warnings()`

These enforce contest constraints (timeouts, etc.). They matter mainly if your agent is slow.

---

## 4. `AgentRules`: Movement, Eating, Death, and Scared Mechanics

`AgentRules` specifies how individual agents interact with the environment.

### 4.1. Legal actions

- `get_legal_actions(state, agent_index)`  
  Uses the agent’s configuration and walls to compute all possible moves via `Actions.get_possible_actions`.

- `filter_for_allowed_actions(possible_actions)`  
  Currently just returns all possible actions unchanged, but is the hook point for additional restrictions.

### 4.2. Applying an action

- `apply_action(state, action, agent_index)`  
  This is the core transition logic for a single agent step. It:
  1. Validates `action` against legal actions.
  2. Updates the agent’s configuration (position) using `Actions.direction_to_vector` and `Configuration.generate_successor`.
  3. If the agent is on a grid point (`nearest_point`), updates whether the agent is a Pacman or ghost:
     - `agent_state.is_pacman` is `True` if exactly one of these is true:
       - The agent is on the red team.
       - The agent’s position is on the red side of the map.
     - In other words, you are Pacman when you are on the opponent’s side.
     - When an agent returns to its own side with `num_carrying > 0`, the carried food is converted into:
       - Score change (`+num_carrying` for red, `-num_carrying` for blue).
       - Increment of `num_returned`.
       - Set `num_carrying` back to 0.
       - If either team’s `num_returned` reaches `(TOTAL_FOOD / 2) - MIN_FOOD`, the game ends.
  4. If the agent is Pacman and close enough to the nearest grid point, calls `consume(nearest, state, is_red)` to eat food or capsules.

### 4.3. Eating food and capsules

- `consume(position, state, is_red)`  
  At a given position:
  - If there is food:
    - Increases `num_carrying` for the Pacman who is exactly at that position.
    - Removes the food from the grid.
    - Records `_food_eaten` for display.
  - If there is a capsule:
    - Removes it from the capsule list.
    - Records `_capsule_eaten`.
    - Sets `scared_timer = SCARED_TIME` for all opponents (the other team).

### 4.4. Scared timer

- `decrement_timer(state)`  
  Decreases the agent’s `scared_timer` by 1 each move. When it reaches 1, the position is snapped to its nearest grid point.

Agents with `scared_timer > 0` behave differently in `check_death` (they can be eaten by Pacmen).

### 4.5. Food dumping on death

- `dump_food_from_death(state, agent_state)`  
  If `DUMP_FOOD_ON_DEATH` is `True` and the dying agent is a Pacman with `num_carrying > 0`, the carried food is redistributed as new pellets via a BFS around the death position, respecting:
  - Map bounds.
  - Walls.
  - Existing food and capsules.
  - Positions of agents.
  - Side correctness (food must be placed on the opponent’s side relative to that dying Pacman’s team).

  `agent_state.num_carrying` is then reset to 0. Newly added food is tracked in `_food_added`.

### 4.6. Death resolution

- `check_death(state, agent_index)`  
  Handles collisions between Pacmen and ghosts:
  - If the agent is a Pacman:
    - Checks all opponents that are ghosts for distance within `COLLISION_TOLERANCE`.
    - If a non-scared ghost catches the Pacman:
      - Optionally dumps carried food (`dump_food_from_death`).
      - Adjusts score by `KILL_POINTS` (possibly negated depending on team).
      - Resets that Pacman to its start position and sets `is_pacman = False` and `scared_timer = 0`.
    - If a scared ghost collides with Pacman:
      - The ghost is sent back to start, `is_pacman = False`, `scared_timer = 0`, and score is adjusted by `KILL_POINTS`.
  - If the agent is a ghost:
    - Symmetric logic but from the ghost perspective: collisions with enemy Pacmen may kill one or the other depending on `scared_timer`.

This defines how combat works and when you should avoid/seek collisions.

- `place_ghost(ghost_state)`  
  Resets a ghost’s configuration to its start position (used internally).

---

## 5. Command-Line Interface and Game Setup

### 5.1. Parsing command-line arguments: `read_command(argv)`

`read_command` uses `optparse.OptionParser` to support many options, including:

- Team selection:
  - `-r, --red`: path to red team Python file (default: `baseline_team` under the contest directory).
  - `-b, --blue`: path to blue team file.
  - `--red-name`, `--blue-name`: display names.
  - `--redOpts`, `--blueOpts`: comma-separated `key=value` options passed to team creation (e.g. `first=keys`).

- Keyboard agents:
  - `--keys0`, `--keys1`, `--keys2`, `--keys3`: replace specific agents with `KeyboardAgent` variants.

- Layout:
  - `-l, --layout`: layout file or `RANDOM` or `RANDOM<seed>` for generated mazes.
  - Only capture layouts are allowed.

- Display options:
  - `-t, --textgraphics`: text-only graphics.
  - `-q, --quiet`: minimal text, no graphics.
  - `-Q, --super-quiet`: also mute agent output.
  - `-z, --zoom`: scaling for graphics.
  - `--delay-step`: delay between moves in play or replay.

- Game parameters:
  - `-i, --time`: time limit (in moves).
  - `-n, --num_games`: number of games to play.
  - `-x, --num_training`: number of training games (no graphics/output).
  - `-c, --catch-exceptions`: enforce timing and crash handling.
  - `-m, --match-identifier`: numeric match identifier.
  - `-u, --contest-name`: contest name used in log paths.

- Randomness:
  - `-f, --fix_random_seed`: fixed seed `"cs188"`.
  - `--setRandomSeed`: set random seed to given string.

- Recording and replay:
  - `--record`: store game histories as replay files.
  - `--record-log`: redirect stdout/stderr to a log file.
  - `--replay`: replay a recorded game with display.
  - `--replayq`: replay a recorded game without graphics to generate logs.

`read_command` also:
- Chooses the display type.
- Loads agents via `load_agents`.
- Inserts keyboard agents where requested.
- Builds the list of layouts (possibly random).
- Packs everything in a dictionary `args` for `run_games`.

Helper:

- `parse_agent_args(input_str)`  
  Parses strings like `"first=keys,foo=bar"` into `{"first": "keys", "foo": "bar"}` to pass to team constructors.

### 5.2. Loading teams: `load_agents(is_red, agent_file, cmd_line_args)`

Steps:

1. Ensures `.py` extension and absolute path.
2. Adds that directory to `sys.path`.
3. Dynamically loads the module via `importlib`.
4. Looks for a function `create_team` defined in that module.
5. Builds agent indices for that team:
   - Red team: indices `[0, 2]`.
   - Blue team: indices `[1, 3]`.
6. Calls `create_team(first_index, second_index, is_red, **args)` and returns the list of agents.

If loading fails or `create_team` is missing, returns `[None, None]`.

As an agent author, you implement `create_team` in your own file, usually something like:

```python
def create_team(first_index, second_index, is_red, **kwargs):
    ...
    return [AgentClass1(first_index), AgentClass2(second_index)]
```

## 6. Running and Recording Games

### 6.1. Random layouts: `random_layout(seed=None)`

Used when the `--layout` option is `RANDOM` or `RANDOM<seed>`.

Behavior:
- If `seed` is not provided, chooses a random integer seed.
- Calls `maze_generator.generate_maze(seed)` to generate a maze.
- Returns a pair `(layout_name, layout_text)` where:
  - `layout_name` is something like `RANDOM12345678`.
  - `layout_text` is the string representation of the maze (multiple lines).

`read_command` uses this to construct a `layout.Layout` object when random layouts are requested.

---

### 6.2. Playing games: `run_games(...)`

Signature (conceptual):

run_games(
    layouts,
    agents,
    display,
    length,
    num_games,
    record,
    num_training,
    red_team_name,
    blue_team_name,
    contest_name="default",
    mute_agents=False,
    catch_exceptions=False,
    delay_step=0,
    match_id=0
)

Key points:

- Creates a single `CaptureRules` instance.
- Iterates `num_games` times:
  - For each game:
    - Picks the corresponding `layout` from `layouts`.
    - If the index `i` is less than `num_training`, treats it as a training game:
      - Replaces `display` with `NullGraphics`.
      - Sets `rules.quiet = True` (no prints).
    - Otherwise uses the given `display` and sets `rules.quiet = False`.
    - Calls `rules.new_game(layout, agents, game_display, length, mute_agents, catch_exceptions)` to construct a `Game`.
    - Calls `g.run(delay=delay_step)` to actually play the game.
    - Collects non-training games into a `games_list`.

- If `record` is `True`:
  - Builds a `components` dict with:
    - `layout`
    - placeholder `agents` (non-functional)
    - `actions` (`g.move_history`),
    - `length`
    - `red_team_name`
    - `blue_team_name`
  - Pickles this dict and writes the replay file to:
    `www/contest_<contest_name>/replays/match_<match_id>.replay`

- After playing:
  - If `num_games > 1`, computes:
    - List of final scores.
    - Red and blue win rates (fraction of scores > 0 or < 0).
    - Prints statistics: average score, scores, win rates, and a record string per game (Red/Blue/Tie).

Returns:
- `games_list` containing the non-training `Game` objects.

---

### 6.3. Replaying games: `replay_game(...)`

Used when `--replay` or `--replayq` is provided.

Parameters include:
- `layout`
- `agents`
- `actions` (list of `(agent_index, action)` pairs)
- `display`
- `length`
- `red_team_name`, `blue_team_name`
- `wait_end` (whether to block at end)
- `delay` (time between steps)

Flow:
- Uses `CaptureRules` and `new_game` to set up an initial game.
- Sets `display.red_team` and `display.blue_team` to the provided names.
- Calls `display.initialize(state.data)`.
- For each `action` in `actions`:
  - Creates a successor state via `state.generate_successor(*action)`.
  - Updates the display (`display.update(state.data)`).
  - Calls `rules.process(state, game)` to check for termination.
  - Sleeps for `delay` seconds between steps.

At the end:
- Marks `game.game_over = True`.
- Prints who won and why (similar logic to `CaptureRules.process`).
- Optionally waits for an ENTER press if `wait_end` is `True`.
- Calls `display.finish()`.

---

## 7. Saving Scores and Statistics

These methods are used after games finish to construct and store match statistics in JSON form.

### 7.1. `get_games_data(games, red_name, blue_name, time_taken, match_id)`

For each `game` in the list:
- Extracts:
  - `layout_name` from `game.state.data.layout.layout_name`.
  - Final `score` from `game.state.data.score`.
- Determines:
  - `winner` as:
    - `red_name` if `score > 0`.
    - `blue_name` if `score < 0`.
    - `None` if `score == 0`.
  - `score` is converted to a non-negative value in the record.
- Appends a tuple:
  (red_name, blue_name, layout_name, score, winner, time_taken, match_id)
Returns the list of these tuples.

---

### 7.2. `compute_team_stats(games_data, team_name)`

Computes a set of statistics for a given `team_name` from `games_data`:

Internal counters:
- `wins`, `draws`, `loses`, `score`.

For each game data tuple (`gd`):
- `gd[4]` is the winner or `None`.
- If `gd[4]` is `None`, increments `draws`.
- Else if `gd[4]` is `team_name`, increments `wins` and adds `gd[3]` (the score) to `score`.
- Otherwise increments `loses`.

Computes:
- `points = wins * 3 + draws`.
- `win_percentage` as `(points * 100) / (3 * total_games)` where `total_games = wins + draws + loses` (or 0 if no games).

Returns a list:
- `[win_percentage, points, wins, draws, loses, 0, score]`
  - The `0` is a placeholder for errors (not tracked here).

---

### 7.3. `save_score(games, total_time, *, contest_name, match_id, **kwargs)`

Assumptions:
- `games` is non-empty; there is at least one completed game.

Steps:
1. Creates the folder:
   `www/contest_<contest_name>/scores`
   if it does not already exist.
2. Calls `get_games_data` with:
   - `games`
   - `red_name` = `kwargs['red_team_name']`
   - `blue_name` = `kwargs['blue_team_name']`
   - `time_taken` = `total_time`
   - `match_id` = `match_id`
3. Computes per-team statistics via `compute_team_stats`:
   - For the red team name.
   - For the blue team name.
4. Creates a `match_data` dictionary with:
   - `'games'`: the `games_data` list.
   - `'max_steps'`: `games[0].length`.
   - `'teams_stats'`: a dict mapping team names to their stats list.
   - `'layouts'`: list of layout names used in `games`.
5. Writes that dictionary as JSON to:
   `www/contest_<contest_name>/scores/match_<match_id>.json`

This is useful for contest infrastructure and post-match analysis, not for agent logic directly.

---

## 8. Top-Level Entry Points: `run` and `main`

### 8.1. `run(args)`

This is the main driver function used when the script is executed.

Flow:
1. Records `start_time`.
2. Calls `read_command(args)` to:
   - Parse command-line options.
   - Set up:
     - `agents`
     - `layouts`
     - `display`
     - `length`
     - `num_games`
     - `num_training`
     - `record`
     - `catch_exceptions`
     - `delay_step`
     - `contest_name`
     - `match_id`
   - Returns them in a dictionary called `options`.
3. Prints `options` to stdout.
4. Calls:
   `games = run_games(**options)`
   to actually play the games.
5. Computes:
   `total_time = round(time.time() - start_time, 0)`
6. If `games` is non-empty:
   - Calls `save_score(games=games, total_time=total_time, **options)`.
7. Prints the total game time.

This function ties together parsing, playing, and saving results.

---

### 8.2. `main()` and script entry

- `main()` simply calls:
  `run(sys.argv[1:])`

- The standard Python pattern:

  if __name__ == '__main__':
      main()

means that when `capture.py` is run directly from the command line, `main()` is executed, which runs the full program as described above.

---

## 9. Summary of What Matters for Agent Authors

From section 6 onward, most logic is about running, replaying, and recording games. As an agent author, you mainly need to know:

- How your team file is loaded:
  - `load_agents` imports your module and calls `create_team(first_index, second_index, is_red, **kwargs)`.
  - Red agents get indices `[0, 2]`, blue agents `[1, 3]`.
- How games are started and repeated:
  - `read_command` sets up everything based on command-line options.
  - `run_games` runs training games (silent) and evaluation games (with statistics).
- How results are recorded:
  - Replays are saved if `--record` is used.
  - JSON summaries of matches and team stats are written by `save_score`.

For agent design and debugging, the critical parts remain:

- The `GameState` interface and what information `make_observation` provides each turn.
- How actions are applied and resolved (`AgentRules.apply_action`, `check_death`, `consume`).
- How scoring, winning, and partial observability work.

Understanding sections 6–9 helps you see the broader environment your agents run in, but the actual behavior of your agents is governed by the earlier GameState and rules logic plus your own team code.
