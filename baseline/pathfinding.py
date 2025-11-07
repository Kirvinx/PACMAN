import heapq

from contest.game import Directions, Actions
from contest.util import manhattan_distance, nearest_point


class AStarPathfinder:
    def __init__(self, agent):
        """
        agent: a CaptureAgent (or subclass) instance.
        Used for get_opponents and get_maze_distance.
        """
        self.agent = agent

    def find_path(self, game_state, start, goal, avoid_enemies=True):
        """
        Find (approximately) optimal path from start to goal position using A*.

        Returns:
            list of actions (Directions.*) to reach goal.
            [] if no path is found.
        """
        walls = game_state.get_walls()

        # Snap start/goal to the nearest grid points
        start = nearest_point(start)
        goal = nearest_point(goal)
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        # Priority queue: (f_score, g_score, position, path)
        h0 = manhattan_distance(start, goal)
        frontier = [(h0, 0, start, [])]

        # Best known g-cost for each position
        best_g = {start: 0}

        width, height = walls.width, walls.height

        while frontier:
            f_score, g_score, current_pos, path = heapq.heappop(frontier)

            # If this state is worse than a previously found path, skip it
            if current_pos in best_g and g_score > best_g[current_pos]:
                continue

            # Goal check
            if current_pos == goal:
                return path

            # Expand neighbors (cardinal directions)
            for action in [Directions.NORTH,
                           Directions.SOUTH,
                           Directions.EAST,
                           Directions.WEST]:
                dx, dy = Actions.direction_to_vector(action)
                nx = current_pos[0] + dx
                ny = current_pos[1] + dy
                next_pos = (nx, ny)

                # Bounds check
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue

                # Wall check
                if walls[nx][ny]:
                    continue

                # Calculate danger penalty if avoiding enemies
                danger_cost = 0
                if avoid_enemies:
                    danger_cost = self.calculate_danger(game_state, next_pos)

                new_g = g_score + 1 + danger_cost
                # If we've already seen a cheaper path to next_pos, skip
                if next_pos in best_g and new_g >= best_g[next_pos]:
                    continue

                best_g[next_pos] = new_g
                h_score = manhattan_distance(next_pos, goal)
                new_f = new_g + h_score

                heapq.heappush(
                    frontier,
                    (new_f, new_g, next_pos, path + [action])
                )

        # No path found
        return []

    def calculate_danger(self, game_state, position):
        """
        Calculate danger level at a position based on enemy proximity.

        Higher returned value = more dangerous = less preferred by A*.
        """
        danger = 0
        enemies = [
            game_state.get_agent_state(i)
            for i in self.agent.get_opponents(game_state)
        ]

        for enemy in enemies:
            enemy_pos = enemy.get_position()
            if enemy_pos is None:
                continue

            # Only treat enemy as dangerous if it is a ghost and not scared
            if not enemy.is_pacman and enemy.scared_timer == 0:
                dist = self.agent.get_maze_distance(position, enemy_pos)
                if dist <= 3:
                    # Heavy penalty for being close to an active ghost
                    danger += (4 - dist) * 10

        return danger
