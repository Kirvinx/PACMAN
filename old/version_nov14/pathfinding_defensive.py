import heapq
import math
from contest.game import Directions, Actions
from contest.util import nearest_point

class DefensiveAStarPathfinder:
    """
    A* pathfinding for defense:
      - Path cost = steps + lambda_ally * repulsion
      - Repulsion is a spatial field centered on *allies* (to spread coverage)
      - Uses exact teammate positions (no beliefs)
      - Cached per timestep, like your offensive version

    Tunables:
      - lambda_ally: weight for avoiding allies
      - repulsion_radius: Manhattan radius of repulsion influence
      - decay_alpha: exponential decay for distance weighting
      - normalize_field: normalize repulsion map to [0, 1]
      - penalize_goal: if False, don't penalize the goal tile (useful if both defenders rally)
    """

    def __init__(self, agent):
        self.agent = agent
        self.cached_repulsion_map = None
        self.cache_timestep = -1

        # Pre-compute direction mappings
        self.directions = [
            (Directions.NORTH, (0, 1)),
            (Directions.SOUTH, (0, -1)),
            (Directions.EAST,  (1, 0)),
            (Directions.WEST,  (-1, 0)),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_path(
            self,
            game_state,
            start,
            goal,
            avoid_allies=True,
            return_cost=False,
            lambda_ally=10.0,
            repulsion_radius=5,
            decay_alpha=0.6,
            normalize_field=True,
            penalize_goal=False,
            ally_positions_override=None,
        ):
            """
            Defensive A*:
                f = g + h
                g = steps + lambda_ally * repulsion(nx, ny)
                h = maze distance heuristic

            Added:
                - HARD MIDLINE BLOCK: defenders never step onto enemy half.
            """
            walls = game_state.get_walls()
            width, height = walls.width, walls.height

            # Normalize start/goal to grid points
            start = nearest_point(start)
            goal = nearest_point(goal)
            start = (int(start[0]), int(start[1]))
            goal = (int(goal[0]), int(goal[1]))

            # Cache repulsion map per timestep
            current_timestep = getattr(game_state.data, "timeleft", 0)
            if avoid_allies and (self.cached_repulsion_map is None or self.cache_timestep != current_timestep):
                self.cached_repulsion_map = self._compute_ally_repulsion_map(
                    game_state,
                    repulsion_radius=repulsion_radius,
                    decay_alpha=decay_alpha,
                    normalize=normalize_field,
                    ally_positions_override=ally_positions_override,
                )
                self.cache_timestep = current_timestep

            repulsion = self.cached_repulsion_map if avoid_allies else None

            # Priority queue: stores (f, g, pos, path)
            frontier = [(self.agent.get_maze_distance(start, goal), 0.0, start, [])]
            best_g = {start: 0.0}
            closed_set = set()

            # Determine midline for hard border constraint
            mid_x = width // 2
            agent_is_red = self.agent.red

            while frontier:
                f, g, pos, path = heapq.heappop(frontier)

                if pos in closed_set:
                    continue
                closed_set.add(pos)

                # Reached goal
                if pos == goal:
                    return (path, g) if return_cost else path

                # Expand neighbors
                for action, (dx, dy) in self.directions:
                    nx, ny = pos[0] + dx, pos[1] + dy

                    # Out of bounds or wall
                    if nx < 0 or nx >= width or ny < 0 or ny >= height or walls[nx][ny]:
                        continue

                    # --- HARD MIDLINE BLOCK ---
                    # Red cannot step into x >= mid_x
                    # Blue cannot step into x < mid_x
                    if agent_is_red:
                        if nx >= mid_x:
                            continue
                    else:
                        if nx < mid_x:
                            continue

                    if (nx, ny) in closed_set:
                        continue

                    # Repulsion from allies
                    repulse_cost = 0.0
                    if avoid_allies and repulsion is not None:
                        if penalize_goal or (nx, ny) != goal:
                            repulse_cost = lambda_ally * repulsion[nx][ny]

                    new_g = g + 1.0 + repulse_cost

                    # If no improvement, skip
                    if (nx, ny) in best_g and new_g >= best_g[(nx, ny)] - 1e-6:
                        continue

                    best_g[(nx, ny)] = new_g

                    # Heuristic
                    h = self.agent.get_maze_distance((nx, ny), goal)

                    heapq.heappush(
                        frontier,
                        (new_g + h, new_g, (nx, ny), path + [action])
                    )

            # No path found
            return ([], None) if return_cost else []


    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_ally_repulsion_map(
        self,
        game_state,
        repulsion_radius=5,
        decay_alpha=0.6,
        normalize=True,
        ally_positions_override=None,
    ):
        """
        Builds a repulsion field centered on allies:
            field[x][y] = sum_over_allies(exp(-alpha * manhattan_dist((x,y), ally)))
        Only legal (non-wall) tiles receive mass.
        """
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        field = [[0.0 for _ in range(height)] for _ in range(width)]
        if repulsion_radius <= 0:
            return field

        # Gather ally positions (excluding self)
        ally_positions = ally_positions_override or self._get_ally_positions(game_state)
        if not ally_positions:
            return field

        # Precompute decay and offsets by distance
        decay_values = [math.exp(-decay_alpha * d) for d in range(repulsion_radius + 1)]
        offset_by_dist = [[] for _ in range(repulsion_radius + 1)]
        for dx in range(-repulsion_radius, repulsion_radius + 1):
            for dy in range(-repulsion_radius, repulsion_radius + 1):
                d = abs(dx) + abs(dy)
                if d <= repulsion_radius:
                    offset_by_dist[d].append((dx, dy))

        # Paint repulsion from each ally
        for ax, ay in ally_positions:
            ax, ay = int(ax), int(ay)
            # Skip if the ally is somehow off-grid/walled
            if not (0 <= ax < width and 0 <= ay < height) or walls[ax][ay]:
                continue

            for d in range(repulsion_radius + 1):
                w = decay_values[d]
                for dx, dy in offset_by_dist[d]:
                    x, y = ax + dx, ay + dy
                    if 0 <= x < width and 0 <= y < height and not walls[x][y]:
                        field[x][y] += w

        # Optional normalization to [0,1]
        if normalize:
            max_val = max((max(col) for col in field), default=0.0)
            if max_val > 0:
                inv = 1.0 / max_val
                for x in range(width):
                    for y in range(height):
                        # Cap to [0,1] in case of numerical noise
                        v = field[x][y] * inv
                        field[x][y] = 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)

        return field

    def _get_ally_positions(self, game_state):
        """
        Returns a list of integer grid positions for teammates (excluding self).
        Tries several common APIs for Capture-the-Flag style projects.
        """
        # 1) Determine team indices
        team_idxs = []
        if hasattr(self.agent, "getTeam"):
            team_idxs = list(self.agent.getTeam(game_state))
        elif hasattr(self.agent, "get_team"):
            team_idxs = list(self.agent.get_team(game_state))
        elif hasattr(self.agent, "get_team_indices"):
            team_idxs = list(self.agent.get_team_indices(game_state))
        else:
            # Fallback: try to infer from opponents if available
            if hasattr(self.agent, "getOpponents"):
                opps = set(self.agent.getOpponents(game_state))
                if hasattr(game_state, "get_num_agents"):
                    all_idx = range(game_state.get_num_agents())
                    team_idxs = [i for i in all_idx if i not in opps]
            # If all else fails, assume only self
            if not team_idxs and hasattr(self.agent, "index"):
                team_idxs = [self.agent.index]

        # 2) Remove self
        my_index = getattr(self.agent, "index", None)
        team_idxs = [i for i in team_idxs if i != my_index]

        # 3) Extract positions
        ally_positions = []
        for i in team_idxs:
            st = game_state.get_agent_state(i)
            pos = None
            # Compatible getters
            if hasattr(st, "get_position"):
                pos = st.get_position()
            elif hasattr(st, "getPosition"):
                pos = st.getPosition()
            elif hasattr(st, "position"):
                pos = st.position
            if pos is None:
                continue
            px, py = nearest_point(pos)
            ally_positions.append((int(px), int(py)))

        return ally_positions