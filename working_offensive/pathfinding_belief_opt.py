import heapq
import math
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

    def find_path(self, game_state, start, goal, avoid_enemies=True, return_cost=False):
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
            self.cached_danger_map, _ = self._compute_belief_danger_fields(game_state)
            self.cache_timestep = current_timestep
        danger_map = self.cached_danger_map if avoid_enemies else None

        # Priority queue: (f, g, pos, path)
        frontier = [(self.agent.get_maze_distance(start, goal), 0.0, start, [])]
        best_g = {start: 0.0}
        closed_set = set()
        lambda_risk = 16.0

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

                # Risk cost from belief-based danger map
                danger_cost = lambda_risk * danger_map[nx][ny] if avoid_enemies and danger_map else 0.0
                new_g = g + 1.0 + danger_cost

                if (nx, ny) in best_g and new_g >= best_g[(nx, ny)] - 1e-6:
                    continue

                best_g[(nx, ny)] = new_g
                h = self.agent.get_maze_distance((nx, ny), goal)
                heapq.heappush(frontier, (new_g + h, new_g, (nx, ny), path + [action]))

        # No path found
        if return_cost:
            return [], None
        else:
            return []

    def _compute_belief_danger_fields(self, game_state, max_manhattan_radius=5):
        """
        Convert ghost beliefs into a spatial danger field in [0,1].
        Only counts ghosts that are actually dangerous
        (self.agent._is_dangerous_ghost == True).
        Optimized version with pre-computed values.
        """
        tracker = getattr(self.agent, "belief_tracker", None)
        assert tracker is not None, "Agent must have a belief_tracker!"

        walls = game_state.get_walls()
        width, height = walls.width, walls.height

        danger_map = [[0.0 for _ in range(height)] for _ in range(width)]
        alpha = 0.5  # decay rate

        # Pre-compute exponential decay values
        decay_values = [math.exp(-alpha * dist) for dist in range(max_manhattan_radius + 1)]

        # Pre-compute valid offsets for each distance
        offset_by_dist = [[] for _ in range(max_manhattan_radius + 1)]
        for dx in range(-max_manhattan_radius, max_manhattan_radius + 1):
            for dy in range(-max_manhattan_radius, max_manhattan_radius + 1):
                dist = abs(dx) + abs(dy)
                if dist <= max_manhattan_radius:
                    offset_by_dist[dist].append((dx, dy))

        # Process beliefs
        for opp, belief in tracker.beliefs.items():
            # Skip ghosts that are not dangerous (e.g. scared)
            opp_state = game_state.get_agent_state(opp)
            if not self.agent._is_dangerous_ghost(opp_state):
                continue

            for (gx, gy), p in belief.items():
                if p <= 1e-6:  # Skip very small probabilities
                    continue

                for dist in range(max_manhattan_radius + 1):
                    decay = p * decay_values[dist]
                    for dx, dy in offset_by_dist[dist]:
                        x, y = gx + dx, gy + dy
                        if 0 <= x < width and 0 <= y < height and not walls[x][y]:
                            danger_map[x][y] += decay

        # Normalize to [0,1]
        max_val = max(max(row) for row in danger_map)
        if max_val > 0:
            inv_max = 1.0 / max_val
            for x in range(width):
                for y in range(height):
                    danger_map[x][y] = min(danger_map[x][y] * inv_max, 1.0)

        return danger_map, None

