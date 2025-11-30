# `layout.py` – Map Structure and Static Board Information

This file defines how **layouts (maps)** are represented and loaded. As an agent author, it tells you:

- How walls, food, capsules, and starting positions are stored.
- How the layout’s size and structure are defined.
- What helper functions exist for random positions and corners.
- How layouts are loaded from `.lay` files.

You usually don’t edit this file, but understanding it helps you reason about positions, walls, and map-specific logic.

---

## 1. The `Layout` Class

A `Layout` object holds all the static information about a game board:

- Maze dimensions (width, height)
- Where walls are
- Where food and capsules are
- Agents’ starting positions
- Number of ghosts

It is created from a text representation of the map (the contents of a `.lay` file).

Constructor:

- `__init__(self, layout_name, layout_text)`
  - `layout_name`: name (e.g. `"defaultCapture"`).
  - `layout_text`: list of strings, each line of the layout file.
  - Sets:
    - `self.width`, `self.height`
    - `self.walls`: Grid(width, height, False)
    - `self.food`: Grid(width, height, False)
    - `self.capsules`: list of capsule positions `(x, y)`
    - `self.agent_positions`: list of `(is_pacman_flag_or_id, (x, y))`
    - `self.num_ghosts`: integer
  - Calls `process_layout_text(layout_text)` to fill these.
  - `self.total_food` = number of positions where food is present.

This `Layout` is then used by `GameStateData.initialize` in `game.py` and by `GameState.initialize` in `capture.py` to create actual game states.

---

## 2. How the Layout Is Parsed

Core method:

- `process_layout_text(self, layout_text)`

It parses a text map where:

- `%` = wall
- `.` = food
- `o` = capsule
- `G` = ghost starting position
- `P` = Pacman starting position
- `1`, `2`, `3`, `4` = numbered ghost positions (for variants)

Important details:

- Coordinates are flipped from input file to internal `(x, y)` convention:
  - Input text: top line is highest row.
  - Internal grid: `(0,0)` is bottom-left.
  - Code loops:
    - `y` from 0 to `height - 1`
    - `layout_char = layout_text[max_y - y][x]` so top of file becomes top of grid.

For each character, it calls:

- `process_layout_char(self, x, y, layout_char)`:

  - If `%`: `self.walls[x][y] = True`
  - If `.`: `self.food[x][y] = True`
  - If `o`: `self.capsules.append((x, y))`
  - If `P`:
    - `self.agent_positions.append((0, (x, y)))`
  - If `G`:
    - `self.agent_positions.append((1, (x, y)))`
    - `self.num_ghosts += 1`
  - If in `['1', '2', '3', '4']`:
    - `self.agent_positions.append((int(layout_char), (x, y)))`
    - `self.num_ghosts += 1`

After reading the whole text:

- `self.agent_positions.sort()`
- Then transforms it to:
  - `self.agent_positions = [(i == 0, pos) for i, pos in self.agent_positions]`
  - This produces a list of tuples `(is_pacman_bool, position)`:
    - For i == 0 → Pacman.
    - For i > 0 → ghosts.

This list is exactly what `GameStateData.initialize` uses to create `AgentState`s (Pacman vs ghost).

---

## 3. Layout Fields and What They Mean

These are the attributes you may want to know about:

- `layout_name`: file or logical name of the layout.
- `width`, `height`: dimensions of the map.
- `walls`: `Grid` of booleans:
  - `walls[x][y] == True` → wall at `(x, y)`.
- `food`: `Grid` of booleans:
  - `food[x][y] == True` → food at `(x, y)`.
- `capsules`: list of positions `(x, y)` where capsules are initially located.
- `agent_positions`: list of `(is_pacman, (x, y))` for each starting agent.
- `num_ghosts`: number of ghost start positions (used in non-capture variants).
- `layout_text`: original text representation of the layout.
- `total_food`: total number of food dots at start (`len(self.food.as_list())`).

Though your agent code usually interacts at the `GameState` level, knowing how this is built helps with debugging or writing layout-specific heuristics.

---

## 4. Visibility Matrix (Optional, Used for Line of Sight)

Method:

- `initialize_visibility_matrix(self)`

This builds `self.visibility`, a `Grid` where each cell contains a mapping:

- For each cell `(x, y)`:
  - `self.visibility[x][y][direction]` is a set of positions visible from `(x, y)` when looking in `direction`.

Rough idea:

- For each non-wall cell `(x, y)`:
  - For each direction (N, S, E, W):
    - Step outward in that direction, adding positions until a wall is encountered.

Visibility matrices are cached in `VISIBILITY_MATRIX_CACHE` keyed by the combined layout text.

Related function:

- `is_visible_from(self, ghost_pos, pac_pos, pac_direction)`:
  - Returns `True` if `ghost_pos` is within the visibility cone of Pacman at `pac_pos` facing `pac_direction`.

This is more relevant to certain variants or visual features; Capture the Flag mostly uses partial observability via noisy distances, but this is good to know is available.

---

## 5. Convenience Methods on Layout

These methods provide useful map-level utilities.

- `get_num_ghosts(self)`:
  - Returns `self.num_ghosts`.

- `is_wall(self, pos)`:
  - `pos` is `(x, y)`.
  - Returns `self.walls[x][y]`.

- `get_random_legal_position(self)`:
  - Picks a random `(x, y)` within the map that is not a wall.
  - Useful for random agent placements or initial testing.

- `get_random_corner(self)`:
  - Returns one of the four inner corners:
    - `(1, 1)`
    - `(1, height - 2)`
    - `(width - 2, 1)`
    - `(width - 2, height - 2)`

- `get_furthest_corner(self, pac_pos)`:
  - Given a position `pac_pos`, picks the corner that maximizes Manhattan distance to `pac_pos`.
  - Uses `manhattan_distance` to compute distances.
  - Useful for “run away to far corner” type behavior.

- `__str__(self)`:
  - Returns the original layout text joined with newlines.
  - Printing a `Layout` instance shows the textual map.

- `deep_copy(self)`:
  - Returns a new `Layout` with the same name and a copy of `layout_text`.

---

## 6. Loading Layouts from Files

There are two helper functions:

1. `get_layout(name, back=2)`

   Logic:

   - If `name` ends with `.lay`:
     - Try `layouts/name`.
     - If not found, try `name` as direct path.
   - Else:
     - Try `layouts/name.lay`.
     - If not found, try `name.lay`.
   - If still not found and `back >= 0`:
     - Move to parent directory, decrease `back`, and try again recursively.
   - Returns a `Layout` object or `None` if not found.

   This allows calling code (like `capture.py`) to say:

   - `layout = layout.get_layout("defaultCapture")`

   without worrying about exact paths.

2. `try_to_load(fullname)`

   - If `fullname` exists:
     - Opens the file.
     - Reads each line, strips it, and builds `layout_text` list.
     - Returns a `Layout` object: `Layout(layout_name=..., layout_text=layout_text)`
   - Else:
     - Returns `None`.

These functions are used by `capture.py` when setting up games, based on CLI arguments like `--layout`.

---

## 7. How This Connects to Your Agent

For writing Capture-the-Flag agents:

- You don’t usually interact with `Layout` directly in agent code, but:
  - `GameState.get_walls()` comes from `self.state.data.layout.walls`.
  - `GameState.get_red_food()` and `get_blue_food()` come from the `Layout`’s `food` grid split in half.
  - `get_capsules()` and team start positions ultimately come from `Layout`.

Understanding `layout.py` helps you:

- Debug map issues (e.g., “Why is there a wall here?”, “Why is total_food weird?”).
- Reason about coordinate systems (top vs bottom, flipped y).
- Potentially write layout-aware heuristics (avoid corners, head for corridors, etc.).

But for most agent logic, you primarily use:

- `game_state.get_walls()`
- `game_state.get_red_food()`, `get_blue_food()`
- `game_state.get_capsules()`
- `get_maze_distance`, `get_legal_actions`, and pathfinding utilities.

`layout.py` is the **static blueprint** that underlies all of that.

