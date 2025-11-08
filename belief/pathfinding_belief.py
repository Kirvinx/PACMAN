import heapq
import math
from contest.game import Directions, Actions
from contest.util import manhattan_distance, nearest_point


class AStarPathfinder:
    def __init__(self, agent):
        self.agent = agent

    def find_path(self, game_state, start, goal, avoid_enemies=True):
        """
        Belief-based A* pathfinding:
        - Path cost = steps + λ * danger
        - Danger derived entirely from belief distributions
        """
        walls = game_state.get_walls()
        start = nearest_point(start)
        goal = nearest_point(goal)
        start, goal = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))

        width, height = walls.width, walls.height
        danger_map, _ = self._compute_belief_danger_fields(game_state)

        # Priority queue: (f, g, pos, path)
        frontier = [(self.agent.get_maze_distance(start, goal), 0.0, start, [])]
        best_g = {start: 0.0}
        lambda_risk = 16.0

        while frontier:
            f, g, pos, path = heapq.heappop(frontier)
            if pos in best_g and g > best_g[pos] + 1e-6:
                continue
            if pos == goal:
                return path

            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.direction_to_vector(action)
                nx, ny = int(pos[0] + dx), int(pos[1] + dy)
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if walls[nx][ny]:
                    continue

                # Risk cost from belief-based danger map
                danger_cost = lambda_risk * danger_map[nx][ny] if avoid_enemies else 0.0
                new_g = g + 1.0 + danger_cost

                if (nx, ny) in best_g and new_g >= best_g[(nx, ny)] - 1e-6:
                    continue

                best_g[(nx, ny)] = new_g
                h = self.agent.get_maze_distance((nx, ny), goal)
                heapq.heappush(frontier, (new_g + h, new_g, (nx, ny), path + [action]))

        return []

    def _compute_belief_danger_fields(self, game_state, max_manhattan_radius=5):
        """
        Convert ghost beliefs into a spatial danger field in [0,1].
        """
        tracker = getattr(self.agent, "belief_tracker", None)
        assert tracker is not None, "Agent must have a belief_tracker!"

        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        legal_positions = tracker.legal_positions or [
            (x, y) for x in range(width) for y in range(height) if not walls[x][y]
        ]

        danger_map = [[0.0 for _ in range(height)] for _ in range(width)]
        alpha = 0.5  # decay rate

        for opp, belief in tracker.beliefs.items():
            for (gx, gy), p in belief.items():
                if p <= 0:
                    continue
                for dx in range(-max_manhattan_radius, max_manhattan_radius + 1):
                    for dy in range(-max_manhattan_radius, max_manhattan_radius + 1):
                        x, y = gx + dx, gy + dy
                        if (
                            0 <= x < width and 0 <= y < height and
                            not walls[x][y] and abs(dx) + abs(dy) <= max_manhattan_radius
                        ):
                            danger_map[x][y] += p * math.exp(-alpha * (abs(dx) + abs(dy)))

        # Normalize to [0,1]
        max_val = max(max(row) for row in danger_map) or 1.0
        for x in range(width):
            for y in range(height):
                danger_map[x][y] = min(danger_map[x][y] / max_val, 1.0)

        return danger_map, None
