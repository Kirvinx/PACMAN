from contest.capture_agents import CaptureAgent
from contest.game import Directions
from behavior_tree import Selector, Sequence, Condition, Action
from pathfinding import AStarPathfinder
import random


class OffensiveAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.pathfinder = AStarPathfinder(self)

        # Behavior Tree
        self.tree = Selector([
            Sequence([
                Condition(self.ghost_nearby),
                Action(self.run_away)
            ]),
            Sequence([
                Condition(self.carrying_too_much),
                Action(self.return_home)
            ]),
            Action(self.hunt_food)
        ])

    def choose_action(self, game_state):
        # The behavior tree decides what to do each turn
        action = self.tree.execute(self, game_state)

        # If nothing returned (shouldn't happen), pick something safe
        if action not in Directions.__dict__.values():
            legal = game_state.get_legal_actions(self.index)
            return random.choice(legal)

        return action

    # ----------------------------
    # Condition Nodes
    # ----------------------------

    def ghost_nearby(self, agent, game_state):
        """Return True if an enemy ghost is close."""
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        my_pos = game_state.get_agent_state(self.index).get_position()

        for e in enemies:
            if e.get_position() and not e.is_pacman:
                dist = self.get_maze_distance(my_pos, e.get_position())
                if dist <= 4:
                    return True
        return False

    def carrying_too_much(self, agent, game_state):
        """Return True if carrying enough food to go home."""
        my_state = game_state.get_agent_state(self.index)
        return my_state.num_carrying >= 3

    # ----------------------------
    # Action Nodes
    # ----------------------------

    def run_away(self, agent, game_state):
        """Find a path away from nearest ghost using A*."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

        ghosts = [e.get_position() for e in enemies if e.get_position() and not e.is_pacman]
        if not ghosts:
            return Directions.STOP

        nearest = min(ghosts, key=lambda g: self.get_maze_distance(my_pos, g))

        # Move away from ghost
        dx = my_pos[0] - nearest[0]
        dy = my_pos[1] - nearest[1]
        target = (my_pos[0] + dx * 2, my_pos[1] + dy * 2)

        path = self.pathfinder.find_path(game_state, my_pos, target)
        return path[0] if path else Directions.STOP

    def return_home(self, agent, game_state):
        """Go back to your side safely using A*."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()
        width = walls.width

        # Determine midline between sides
        if self.red:
            home_x = (width // 2) - 1
        else:
            home_x = (width // 2)

        # Find nearest point on your side of midline that’s not a wall
        home_targets = [
            (home_x, y)
            for y in range(walls.height)
            if not walls[home_x][y]
        ]

        nearest = min(home_targets, key=lambda p: self.get_maze_distance(my_pos, p))
        path = self.pathfinder.find_path(game_state, my_pos, nearest)
        return path[0] if path else Directions.STOP

    def hunt_food(self, agent, game_state):
        """Move toward the nearest food using A*."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        food_list = self.get_food(game_state).as_list()

        if not food_list:
            return Directions.STOP

        nearest_food = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        path = self.pathfinder.find_path(game_state, my_pos, nearest_food)
        return path[0] if path else Directions.STOP
