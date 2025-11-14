from collections import deque
import math


class TerritoryAnalyzer:
    def __init__(self, game_state, is_red):
        """
        Static map + entrance analysis for defensive positioning.

        - Detects a structurally good 'entrance column' based on walls.
        - Defines entrances as all free tiles in that column.
        - Precomputes BFS distances from EACH entrance to ALL tiles:
              distances_to_entrances[e_idx][x][y]
        """
        self.walls = game_state.get_walls()
        self.food = game_state.get_red_food() if is_red else game_state.get_blue_food()
        self.is_red = is_red
        self.width = self.walls.width
        self.height = self.walls.height
        self.mid_x = self.width // 2

        # Not strictly needed right now, but kept for clarity
        self.home_side = range(0, self.mid_x) if is_red else range(self.mid_x, self.width)
        self.enemy_side = range(self.mid_x, self.width) if is_red else range(0, self.mid_x)

        # Entrances and distances
        self.entrances = self._compute_entrances()
        self.distances_to_entrances = self._precompute_distances()

    # ------------------------------------------------------------------
    # Entrance detection
    # ------------------------------------------------------------------
    def _compute_entrances(self):
        """
        Compute entrances based on vertical wall density:

        1. Find the food on our side that is closest to enemy side (furthest along x).
        2. Scan all vertical columns between that food's x and the border column (inclusive).
        3. For each column, count how many tiles are walls.
        4. Choose the column with the MOST walls.
        5. Entrances = all non-wall tiles in that chosen column.
        """
        food_list = self.food.as_list()
        if not food_list:
            return []

        # 1. Find furthest food toward enemy
        furthest_food = max(
            food_list,
            key=lambda pos: pos[0] if self.is_red else -pos[0]
        )
        fx, fy = furthest_food

        # 2. Determine border column for this side
        border_x = self.mid_x - 1 if self.is_red else self.mid_x

        # Clamp fx and border_x inside map bounds
        fx = max(0, min(fx, self.width - 1))
        border_x = max(0, min(border_x, self.width - 1))

        # 3. Define the scan range (always low → high, independent of side)
        start_x = min(fx, border_x)
        end_x = max(fx, border_x)

        # 4. Scan each column and count walls
        column_scores = []  # list of (x, wall_count)
        for x in range(start_x, end_x + 1):
            if not (0 <= x < self.width):
                continue
            wall_count = sum(1 for y in range(self.height) if self.walls[x][y])
            column_scores.append((x, wall_count))

        # 5. Fallback if something went weird and we scanned nothing
        if not column_scores:
            entrance_x = border_x
        else:
            # Pick column with highest wall count
            entrance_x, _ = max(column_scores, key=lambda item: item[1])

        # 6. Entrances are simply all non-wall positions in that column
        entrances = [
            (entrance_x, y)
            for y in range(self.height)
            if not self.walls[entrance_x][y]
        ]

        return entrances

    # ------------------------------------------------------------------
    # Distance tensor: entrances x width x height
    # ------------------------------------------------------------------
    def _precompute_distances(self):
        """
        Precompute BFS distance from EACH entrance to ALL reachable tiles.

        Returns:
            distances: list of 2D arrays
                distances[e_idx][x][y] = distance from self.entrances[e_idx] to (x, y),
                                         or math.inf if unreachable.
        """
        num_entrances = len(self.entrances)
        if num_entrances == 0:
            return []

        distances = []

        for (sx, sy) in self.entrances:
            # Initialize distance grid for this entrance
            dist = [[math.inf for _ in range(self.height)] for _ in range(self.width)]

            # BFS from this entrance
            q = deque()
            dist[sx][sy] = 0
            q.append((sx, sy))

            while q:
                x, y = q.popleft()
                d = dist[x][y]

                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy

                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    if self.walls[nx][ny]:
                        continue
                    if dist[nx][ny] <= d + 1:
                        continue

                    dist[nx][ny] = d + 1
                    q.append((nx, ny))

            distances.append(dist)

        return distances

    # ------------------------------------------------------------------
    # Enemy distances to each entrance (from ACTUAL positions)
    # ------------------------------------------------------------------
    def _compute_enemy_distances_to_entrances(self, enemy_positions):
        """
        enemy_positions: list of (ex, ey) for enemies (actual known positions).
        Returns: list d_opp where d_opp[i] is the MIN distance from any enemy to entrance i.
        """
        num_entrances = len(self.entrances)
        d_opp = [math.inf] * num_entrances

        if not enemy_positions:
            return d_opp  # no enemies visible -> treat as very far

        for i in range(num_entrances):
            dist_grid = self.distances_to_entrances[i]  # 2D array [x][y]
            best = math.inf
            for (ex, ey) in enemy_positions:
                if not (0 <= ex < self.width and 0 <= ey < self.height):
                    continue
                d = dist_grid[ex][ey]
                if d < best:
                    best = d
            d_opp[i] = best

        return d_opp

    # ------------------------------------------------------------------
    # Lexicographic entrance-priority heuristic
    # ------------------------------------------------------------------
    def entrance_priority_score(self, defender_pos, enemy_positions):
        """
        defender_pos: (dx, dy)
        enemy_positions: list[(ex, ey)] of ACTUAL enemy positions (not beliefs).

        Heuristic:

        - For each entrance, in order of how close it is for the ENEMY:
            * If defender distance <= enemy distance -> score_i = 0
              (we're early enough; extra closeness doesn't matter)
            * Else score_i = defender_distance - enemy_distance
        - Return the vector [score_0, score_1, ..., score_k-1].
        - Use lexicographic minimization over this vector.
        """
        dx, dy = defender_pos
        num_entrances = len(self.entrances)
        if num_entrances == 0:
            return ()  # no entrances -> everything is equal

        if not enemy_positions:
            # No enemies to defend against -> all tiles equal
            return ()

        # 1) enemy distance to each entrance
        d_opp = self._compute_enemy_distances_to_entrances(enemy_positions)

        # 2) defender distance to each entrance (using precomputed tensor)
        d_def = [self.distances_to_entrances[i][dx][dy] for i in range(num_entrances)]

        # 3) sort entrances by enemy closeness (smallest d_opp first)
        indices = list(range(num_entrances))
        indices.sort(key=lambda i: d_opp[i])

        # 4) build lexicographic score vector
        scores = []
        for i in indices:
            di_def = d_def[i]
            di_opp = d_opp[i]

            if di_def <= di_opp:
                scores.append(0)  # we are on-time or earlier
            else:
                scores.append(di_def - di_opp)  # how late we are vs enemy

        return tuple(scores)

    # ------------------------------------------------------------------
    # Choosing the best defensive tile
    # ------------------------------------------------------------------
    def choose_best_defensive_tile(self, candidate_tiles, enemy_positions):
        """
        candidate_tiles: list of (x, y) tiles you are willing to stand on.
                         e.g., all home tiles near border, or a guard band.
        enemy_positions: list of ACTUAL enemy positions (ex, ey).
                         If empty, we don't pick any "best" tile.

        Returns:
            The (x, y) tile that best satisfies the lexicographic entrance heuristic,
            or None if there are no candidates or no enemies.
        """
        if not candidate_tiles or not enemy_positions:
            return None

        return min(
            candidate_tiles,
            key=lambda pos: self.entrance_priority_score(pos, enemy_positions)
        )
