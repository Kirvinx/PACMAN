from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import manhattan_distance
from contest.agents.team_name_1.beliefline.pathfinding_belief import AStarPathfinder
from contest.agents.team_name_1.beliefline.belief_system import GhostBeliefTracker
import random


class BeliefOffensiveAgent(CaptureAgent):
    """
    Simple offensive agent wired to:
      - GhostBeliefTracker (probabilistic ghost locations)
      - Belief-based A* pathfinding (danger-aware)

    Logic:
      - Update beliefs (elapse_time + observe) every turn.
      - If a ghost is probably nearby on enemy side: run away toward home.
      - Else if carrying enough food: return home.
      - Else: go to nearest food.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # Belief system over enemy positions
        self.belief_tracker = GhostBeliefTracker(self, self.get_opponents(game_state))
        self.belief_tracker.initialize_uniformly(game_state)

        # Belief-aware A*
        self.pathfinder = AStarPathfinder(self)

    def choose_action(self, game_state):
        # --- 1) Update beliefs first ---
        # Predict ghost motion
        self.belief_tracker.elapse_time(game_state)
        # Incorporate new noisy distances + visibility
        self.belief_tracker.observe(game_state)

        # Optional: visualize beliefs for debugging
        # self.display_distributions_over_positions(
        #     [self.belief_tracker.beliefs[i] for i in self.get_opponents(game_state)]
        # )

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Which side are we on?
        width = game_state.data.layout.width
        mid_x = width // 2
        if self.red:
            on_enemy_side = my_pos[0] >= mid_x
        else:
            on_enemy_side = my_pos[0] < mid_x

        # 1) Ghost check using beliefs (only when on enemy side)
        if on_enemy_side and self.ghost_nearby_belief(game_state, radius=4, prob_threshold=0.2):
            print("Belief: Run\n")
            return self.run_away(game_state)

        # 2) Food carrying check
        if my_state.num_carrying >= 10:
            return self.return_home(game_state)

        # 3) Default: hunt food
        print("Belief: Hunt\n")
        return self.hunt_food(game_state)

    # ----------------------------
    # Belief-based helper
    # ----------------------------

    def ghost_nearby_belief(self, game_state, radius=4, prob_threshold=0.2):
        """
        Return True if, according to our beliefs, there is at least
        `prob_threshold` probability that some enemy ghost is within
        `radius` maze distance of us.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return False

        total_close_prob = 0.0

        for opp in self.get_opponents(game_state):
            belief = self.belief_tracker.beliefs.get(opp)
            if not belief:
                continue

            # Sum probability mass of ghost positions within radius
            for pos, p in belief.items():
                if p <= 0.0:
                    continue
                # Use maze distance for accuracy; you could approximate with Manhattan for speed
                dist = self.get_maze_distance(my_pos, pos)
                if dist <= radius:
                    total_close_prob += p
                    if total_close_prob >= prob_threshold:
                        return True

        return False

    def is_on_enemy_side(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()
        width = walls.width
        if self.red:
            return my_pos[0] >= width // 2
        else:
            return my_pos[0] < width // 2

    # ----------------------------
    # Actions (same structure as your simple agent)
    # ----------------------------

    def run_away(self, game_state):
        """
        Head toward home boundary using belief-based A* (avoid_enemies=True).
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()
        width = walls.width

        if self.red:
            home_x = (width // 2) - 1
        else:
            home_x = (width // 2)

        home_targets = [(home_x, y) for y in range(walls.height) if not walls[home_x][y]]
        if not home_targets:
            return Directions.STOP

        target = min(home_targets, key=lambda p: self.get_maze_distance(my_pos, p))

        path = self.pathfinder.find_path(
            game_state,
            my_pos,
            target,
            avoid_enemies=True
        )

        if path:
            return path[0]

        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP
        return random.choice(legal)

    def return_home(self, game_state):
        """
        Same as run_away, but used when carrying food.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()
        width = walls.width

        if self.red:
            home_x = (width // 2) - 1
        else:
            home_x = (width // 2)

        home_targets = [(home_x, y) for y in range(walls.height) if not walls[home_x][y]]
        if not home_targets:
            return Directions.STOP

        target = min(home_targets, key=lambda p: self.get_maze_distance(my_pos, p))

        path = self.pathfinder.find_path(
            game_state,
            my_pos,
            target,
            avoid_enemies=True
        )

        if path:
            return path[0]

        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP
        return random.choice(legal)

    def hunt_food(self, game_state):
        """
        Go to the nearest food using belief-aware A*.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        food = self.get_food(game_state).as_list()

        if not food:
            return Directions.STOP

        target = min(food, key=lambda f: self.get_maze_distance(my_pos, f))

        path = self.pathfinder.find_path(
            game_state,
            my_pos,
            target,
            avoid_enemies=True  # flip to False to compare behavior
        )

        if path:
            return path[0]

        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP
        return random.choice(legal)
