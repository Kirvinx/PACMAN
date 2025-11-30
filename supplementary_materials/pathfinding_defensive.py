import heapq
from contest.game import Directions, Actions
from contest.util import nearest_point

class DefensiveAStarPathfinder:
    """
    A* pathfinding for defense with leader-follower coordination:
      - Path cost = steps + lambda_path_overlap * overlap_penalty
      - Uses path tile avoidance for teammate coordination
      - Enforces hard midline boundary (defenders stay on their side)
      - Cached per timestep for efficiency
    """

    def __init__(self, agent):
        self.agent = agent
        self.cache_timestep = -1
        
        # Pre-compute direction mappings
        self.directions = [
            (Directions.NORTH, (0, 1)),
            (Directions.SOUTH, (0, -1)),
            (Directions.EAST,  (1, 0)),
            (Directions.WEST,  (-1, 0)),
        ]

    def find_path(
        self,
        game_state,
        start,
        goal,
        return_cost=False,
        avoid_path_tiles=None,
        lambda_path_overlap=100.0,  # High penalty for overlapping teammate paths
        penalize_goal=False,
    ):
        """
        Defensive A* with path tile avoidance:
            f = g + h
            g = steps + lambda_path_overlap * (1 if tile in avoid_path_tiles else 0)
            h = maze distance heuristic
            
        Includes HARD MIDLINE BLOCK: defenders never step onto enemy half.
        
        Args:
            game_state: Current game state
            start: Starting position
            goal: Goal position
            return_cost: If True, return (path, cost) tuple
            avoid_path_tiles: Set of (x,y) tiles to avoid (teammate's predicted path)
            lambda_path_overlap: Penalty weight for stepping on avoided tiles
            penalize_goal: If False, don't penalize the goal tile even if in avoid_path_tiles
        """
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        
        # Normalize start/goal to grid points
        start = nearest_point(start)
        goal = nearest_point(goal)
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
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
                # Red defenders cannot step into x >= mid_x
                # Blue defenders cannot step into x < mid_x
                if agent_is_red:
                    if nx >= mid_x:
                        continue
                else:
                    if nx < mid_x:
                        continue
                
                if (nx, ny) in closed_set:
                    continue
                
                # Calculate overlap cost for stepping on teammate's predicted path
                overlap_cost = 0.0
                if avoid_path_tiles is not None and (nx, ny) in avoid_path_tiles:
                    # Don't penalize goal if configured not to
                    if penalize_goal or (nx, ny) != goal:
                        overlap_cost = lambda_path_overlap
                
                new_g = g + 1.0 + overlap_cost
                
                # If no improvement, skip
                if (nx, ny) in best_g and new_g >= best_g[(nx, ny)] - 1e-6:
                    continue
                
                best_g[(nx, ny)] = new_g
                
                # Heuristic (maze distance)
                h = self.agent.get_maze_distance((nx, ny), goal)
                
                heapq.heappush(
                    frontier,
                    (new_g + h, new_g, (nx, ny), path + [action])
                )
        
        # No path found
        return ([], None) if return_cost else []