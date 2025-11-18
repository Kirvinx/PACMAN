import heapq
import math
from collections import deque  # <-- NEW
from contest.game import Directions, Actions
from contest.util import manhattan_distance, nearest_point


class AStarPathfinder:
    def __init__(self, agent):
        self.agent = agent
        self.cached_danger_map = None
        self.cache_timestep = -1
        # Pre-compute direction mappings
        self.directions = [
            (Directions.NORTH, (0, 1)),
            (Directions.SOUTH, (0, -1)),
            (Directions.EAST, (1, 0)),
            (Directions.WEST, (-1, 0))
        ]

    def find_path(self, game_state, start, goal, avoid_enemies=True, return_cost=False, avoid_instant_death=True):
        """
        Belief-based A* pathfinding:
        - Path cost = steps + λ * danger
        - Danger derived entirely from belief distributions

        If return_cost is True, returns (path, cost).
        Otherwise returns path only (backwards compatible).
        """
        walls = game_state.get_walls()
        start = nearest_point(start)
        goal = nearest_point(goal)
        start, goal = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))

        width, height = walls.width, walls.height

        # Cache danger map per timestep
        current_timestep = game_state.data.timeleft if hasattr(game_state.data, 'timeleft') else 0
        if avoid_enemies and (self.cached_danger_map is None or self.cache_timestep != current_timestep):
            # default: max_radius=5, top_k_sources=3
            self.cached_danger_map, _ = self._compute_belief_danger_fields(game_state)
            self.cache_timestep = current_timestep
        danger_map = self.cached_danger_map if avoid_enemies else None

        suicide_tiles = None
        if avoid_enemies and avoid_instant_death:
            suicide_tiles = self._compute_instant_death_tiles(game_state)

        # Priority queue: (f, g, pos, path)
        frontier = [(self.agent.get_maze_distance(start, goal), 0.0, start, [])]
        best_g = {start: 0.0}
        closed_set = set()
        lambda_risk = 64.0

        while frontier:
            f, g, pos, path = heapq.heappop(frontier)

            if pos in closed_set:
                continue
            closed_set.add(pos)

            if pos == goal:
                if return_cost:
                    return path, g
                else:
                    return path

            for action, (dx, dy) in self.directions:
                nx, ny = int(pos[0] + dx), int(pos[1] + dy)

                if nx < 0 or nx >= width or ny < 0 or ny >= height or walls[nx][ny]:
                    continue

                if (nx, ny) in closed_set:
                    continue

                if suicide_tiles is not None and suicide_tiles[nx][ny]:
                    continue

                # Risk cost from belief-based danger map
                danger_cost = lambda_risk * danger_map[nx][ny] if avoid_enemies and danger_map else 0.0
                new_g = g + 1.0 + danger_cost

                if (nx, ny) in best_g and new_g >= best_g[(nx, ny)] - 1e-6:
                    continue

                best_g[(nx, ny)] = new_g
                h = self.agent.get_maze_distance((nx, ny), goal)
                heapq.heappush(frontier, (new_g + h, new_g, (nx, ny), path + [action]))

        print("no_path_found")
        # No path found
        if return_cost:
            return [], None
        else:
            return []

    def _compute_belief_danger_fields(self, game_state, max_manhattan_radius=5, top_k_sources=1):
        """
        Build a danger map from ghost beliefs:

        - For each opponent ghost:
          - If visible: mark its exact tile and neighbors with high danger.
          - From its belief distribution, take up to `top_k_sources` most likely positions.
          - If the max belief is ~1.0, only use that single position (we're "certain").
          - From each source position, BFS through the maze (respecting walls) up to
            `max_manhattan_radius` steps and add decayed danger.

        Returns:
            danger_map, None
        """
        tracker = getattr(self.agent, "belief_tracker", None)
        assert tracker is not None, "Agent must have a belief_tracker!"

        walls = game_state.get_walls()
        width, height = walls.width, walls.height

        danger_map = [[0.0 for _ in range(height)] for _ in range(width)]

        # Decay per maze step
        max_radius = max_manhattan_radius
        alpha = 0.5
        decay_values = [math.exp(-alpha * dist) for dist in range(max_radius + 1)]

        for opp, belief in tracker.beliefs.items():
            opp_state = game_state.get_agent_state(opp)
            visible_pos = opp_state.get_position()
            ghost_visible = visible_pos is not None

            # Only consider dangerous ghosts
            if not self.agent._is_dangerous_ghost(opp_state):
                continue

            # -------------------------------
            # VISIBLE GHOST → HARD DANGER
            # -------------------------------
            if ghost_visible:
                vx, vy = int(visible_pos[0]), int(visible_pos[1])

                if 0 <= vx < width and 0 <= vy < height:
                    # Exact ghost position is max danger
                    danger_map[vx][vy] = max(danger_map[vx][vy], 1.0)

                    # Immediate neighbors also very dangerous
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = vx + dx, vy + dy
                        if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
                            danger_map[nx][ny] = max(danger_map[nx][ny], 0.9)

            # -------------------------------------------------------
            # BELIEF distribution → maze-distance BFS spreading
            # -------------------------------------------------------
            if not belief:
                continue

            items = list(belief.items())
            # Skip if everything is basically zero
            if not items:
                continue

            # Find max-probability position
            (max_pos_f, max_p) = max(items, key=lambda kv: kv[1])
            if max_p <= 1e-6:
                continue

            # If we are *certain* (belief ~ 1), only use that single position
            certainty_threshold = 0.999
            if max_p >= certainty_threshold:
                source_positions = [(max_pos_f, max_p)]
            else:
                # Otherwise, take top-K most likely positions for this enemy
                sorted_items = sorted(items, key=lambda kv: -kv[1])
                source_positions = [(pos_f, p) for (pos_f, p) in sorted_items[:top_k_sources] if p > 1e-6]

            for (pos_f, p) in source_positions:
                gx, gy = int(pos_f[0]), int(pos_f[1])

                if not (0 <= gx < width and 0 <= gy < height):
                    continue
                if walls[gx][gy]:
                    continue

                # If ghost is visible, treat these as near-certain danger;
                # otherwise use probability-amplified p^3 as before.
                effective_p = 1.0 if ghost_visible else (p ** 3)

                # BFS from (gx, gy) limited to max_radius steps (maze distance)
                queue = deque()
                queue.append((gx, gy, 0))
                visited = set()
                visited.add((gx, gy))

                while queue:
                    x, y, dist = queue.popleft()
                    if dist > max_radius:
                        continue

                    decay = effective_p * decay_values[dist]
                    danger_map[x][y] += decay

                    if dist == max_radius:
                        continue

                    for _, (dx, dy) in self.directions:
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < width and 0 <= ny < height and
                            not walls[nx][ny] and
                            (nx, ny) not in visited
                        ):
                            visited.add((nx, ny))
                            queue.append((nx, ny, dist + 1))

        # Normalize to [0, 1]
        max_val = max(max(row) for row in danger_map)
        if max_val > 0:
            inv = 1.0 / max_val
            for x in range(width):
                for y in range(height):
                    danger_map[x][y] = min(danger_map[x][y] * inv, 1.0)

        return danger_map, None

    def _compute_instant_death_tiles(self, game_state, max_radius=1):
        """
        Returns a boolean grid suicide_tiles[x][y] = True if moving onto (x, y)
        could lead to death in 1 ghost move (given current ghost positions).

        Currently uses only *visible* dangerous ghosts.
        You can extend this with beliefs if you want.
        """
        walls = game_state.get_walls()
        width, height = walls.width, walls.height

        suicide_tiles = [[False for _ in range(height)] for _ in range(width)]

        tracker = getattr(self.agent, "belief_tracker", None)
        assert tracker is not None, "Agent must have a belief_tracker!"

        for opp, belief in tracker.beliefs.items():
            opp_state = game_state.get_agent_state(opp)

            # Only consider dangerous ghosts
            if not self.agent._is_dangerous_ghost(opp_state):
                continue

            ghost_pos = opp_state.get_position()
            if ghost_pos is None:
                # Ghost not visible – we could optionally use beliefs here,
                # but for now we skip it to avoid over-blocking.
                continue

            gx, gy = int(ghost_pos[0]), int(ghost_pos[1])
            if not (0 <= gx < width and 0 <= gy < height) or walls[gx][gy]:
                continue

            # BFS from the ghost up to max_radius (normally 1)
            from collections import deque
            queue = deque()
            queue.append((gx, gy, 0))
            visited = {(gx, gy)}

            while queue:
                x, y, dist = queue.popleft()
                if dist > max_radius:
                    continue

                # Any tile within this radius is suicide to step onto
                suicide_tiles[x][y] = True

                if dist == max_radius:
                    continue

                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < width and 0 <= ny < height and
                        not walls[nx][ny] and
                        (nx, ny) not in visited
                    ):
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))

        return suicide_tiles
