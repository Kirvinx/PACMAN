from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import manhattan_distance
from contest.agents.team_name_1.beliefline.pathfinding_belief_opt import AStarPathfinder
from contest.agents.team_name_1.beliefline.belief_system_opt import GhostBeliefTracker
from contest.agents.team_name_1.baseline.behavior_tree import (
    Selector,
    Sequence,
    Condition,
    Action,
)
from contest.agents.team_name_1.beliefline.topology import MapTopologyAnalyzer

import random


class BeliefBTOffensiveAgent(CaptureAgent):
    """
    Offensive agent using:
      - GhostBeliefTracker (probabilistic ghost locations)
      - Belief-based A* pathfinding
      - Behavior Tree for high-level decisions

    Behavior Tree (simple version):

        Selector
         ├── Sequence "Emergency Retreat"
         │    ├── Condition: on enemy side AND ghost nearby (belief)
         │    └── Action: run_away
         ├── Sequence "Return Home With Food"
         │    ├── Condition: carrying >= CARRY_THRESHOLD
         │    └── Action: return_home
         └── Action: hunt_food
    """

    CARRY_THRESHOLD = 10  # tweak later

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # Topology analyzer on static walls
        self.topology = MapTopologyAnalyzer(game_state.get_walls())

        # Belief system over enemy positions
        self.belief_tracker = GhostBeliefTracker(self, self.get_opponents(game_state))
        self.belief_tracker.initialize_uniformly(game_state)

        # Belief-aware A*
        self.pathfinder = AStarPathfinder(self)

        # --- Build Behavior Tree ---

        def cond_ghost_threat(agent, gs):
            # only care when on enemy side
            return agent.is_on_enemy_side(gs) and agent.ghost_nearby(
                gs, radius=4
            )

        def cond_carrying_too_much(agent, gs):
            my_state = gs.get_agent_state(agent.index)
            return my_state.num_carrying >= agent.CARRY_THRESHOLD

        self.behavior_tree = Selector([
            # 1) Emergency retreat if ghost nearby on enemy side
            Sequence([
                Condition(cond_ghost_threat),
                Action(lambda agent, gs: agent.run_away(gs)),
            ]),
            # 2) Return home if carrying too much food
            Sequence([
                Condition(cond_carrying_too_much),
                Action(lambda agent, gs: agent.return_home(gs)),
            ]),
            # 3) Default behavior: hunt food
            Action(lambda agent, gs: agent.hunt_food(gs)),
        ])

    def _choose_escape_food_target(self, game_state, my_pos, food_list):
        """
        When running away but not committing to go home:
        choose a new food target at random, with higher probability
        for food that is farther away from our current position.
        """
        if not food_list:
            return None

        # Compute distances and weights (farther -> bigger weight)
        entries = []
        for f in food_list:
            d = self.get_maze_distance(my_pos, f)
            # Prevent zero; emphasize farther food (quadratic)
            w = max(1, d) ** 2
            entries.append((f, w))

        total_w = sum(w for _, w in entries)
        if total_w <= 0:
            # Fallback: uniform random
            return random.choice(food_list)

        r = random.uniform(0, total_w)
        acc = 0.0
        for f, w in entries:
            acc += w
            if r <= acc:
                return f

        # Fallback, should not happen
        return entries[-1][0]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def is_on_enemy_side(self, game_state):
        """
        True if we're on the opponent's half of the map.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return False

        mid_x = game_state.get_walls().width // 2
        if self.red:
            # red starts on left
            return my_pos[0] >= mid_x
        else:
            # blue starts on right
            return my_pos[0] < mid_x

    def get_believed_ghost_positions(self, game_state):
        """
        Returns a list of 'most likely' ghost positions based on beliefs.
        Falls back to visible positions if needed.
        """
        positions = []
        for opp in self.get_opponents(game_state):
            # assuming tracker has get_belief(agent_index) -> dict[pos] = prob
            belief_dist = self.belief_tracker.get_belief(opp)
            if belief_dist:
                best_pos = max(belief_dist.items(), key=lambda kv: kv[1])[0]
                positions.append(best_pos)
            else:
                opp_state = game_state.get_agent_state(opp)
                opp_pos = opp_state.get_position()
                if opp_pos is not None:
                    positions.append(opp_pos)
        return positions

    # ------------------------------------------------------------------
    # Behavior tree entry
    # ------------------------------------------------------------------

    def choose_action(self, game_state):
        # --- 1) Update beliefs first ---
        self.belief_tracker.elapse_time(game_state)
        self.belief_tracker.observe(game_state)

        # --- 2) Let the behavior tree decide ---
        action = self.behavior_tree.execute(self, game_state)

        # Safety: if BT returns something weird, fall back
        legal = game_state.get_legal_actions(self.index)
        if action in legal:
            return action
        if legal:
            return random.choice(legal)
        return Directions.STOP

    # ------------------------------------------------------------------
    # Ghost threat check (true positions only)
    # ------------------------------------------------------------------

    def ghost_nearby(self, game_state, radius=5):
        """
        Uses *true* ghost positions only (no beliefs).
        Returns True if any visible enemy ghost is within `radius` maze distance.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return False

        for opp in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp)
            opp_pos = opp_state.get_position()

            # Only consider visible ghosts (not Pacmen)
            if opp_pos is not None and not opp_state.is_pacman:
                dist = self.get_maze_distance(my_pos, opp_pos)
                if dist <= radius:
                    return True

        return False

    # ------------------------------------------------------------------
    # Food selection / hunting
    # ------------------------------------------------------------------

    def _score_food_home_side(self, game_state, food_pos, food_list, believed_ghosts):
        """
        Score for food when we are on OUR side.
        Higher is better.
        Components:
          - cluster bonus  (more food nearby)
          - ghost distance bonus (farther from ghosts)
          - risk penalty (deeper trap -> worse)
        """
        # Cluster: count food within radius 3
        cluster_radius = 3
        cluster_size = sum(
            1 for f in food_list if manhattan_distance(food_pos, f) <= cluster_radius
        ) - 1  # exclude self

        # Risk from topology
        risk_depth = self.topology.trap_depth(food_pos)

        # Believed ghost distance
        if believed_ghosts:
            ghost_dist = min(
                self.get_maze_distance(food_pos, gpos) for gpos in believed_ghosts
            )
        else:
            ghost_dist = 10  # neutral if no belief

        # Weights (tune later)
        w_cluster = 3.0
        w_ghost = 1.0
        w_risk = 2.0

        score = (
            w_cluster * cluster_size +
            w_ghost * ghost_dist -
            w_risk * risk_depth
        )
        return score

    def _choose_food_enemy_side(self, game_state, my_pos, food_list):
        """
        Deterministic: pick nearest food by maze distance.
        Used when we are on ENEMY side or as fallback.
        """
        if not food_list:
            return None
        return min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))

    def _choose_food_home_side(self, game_state, my_pos, food_list):
        """
        Biased random choice among good candidates on OUR side:
          - favors clusters
          - avoids deep traps
          - favors being far from believed ghosts
        """
        if not food_list:
            return None

        believed_ghosts = self.get_believed_ghost_positions(game_state)
        scored = [
            (self._score_food_home_side(game_state, f, food_list, believed_ghosts), f)
            for f in food_list
        ]

        # Sort by score (descending)
        scored.sort(reverse=True, key=lambda x: x[0])

        # Sample randomly from top-K
        k = min(5, len(scored))
        top_candidates = [f for _, f in scored[:k]]
        return random.choice(top_candidates)

    def _pick_food_target(self, game_state, my_pos, food_list):
        """
        Wrapper that chooses food depending on which side we're on.
        """
        if not food_list:
            return None

        if self.is_on_enemy_side(game_state):
            return self._choose_food_enemy_side(game_state, my_pos, food_list)
        else:
            return self._choose_food_home_side(game_state, my_pos, food_list)

    def _visible_ghost_positions(self, game_state):
        positions = []
        for opp in self.get_opponents(game_state):
            st = game_state.get_agent_state(opp)
            pos = st.get_position()
            if pos is not None and not st.is_pacman:
                positions.append(pos)
        return positions

    def _safe_to_enter_trap_for_food(self, game_state, my_pos, food_pos):
        """
        Called when we are close to the entrance of a trap region for a target food.

        Rule: if there are visible ghosts, each must be farther than 2 * risk + 1
        from the entrance door, otherwise it's unsafe to enter.
        """
        if not self.topology.is_in_trap_region(food_pos):
            return True

        pocket = self.topology.pocket_id.get(food_pos)
        if pocket is None:
            return True

        exits = list(self.topology.pocket_exits[pocket])
        if not exits:
            # completely enclosed: treat as unsafe
            return False

        # Choose nearest door from our current position
        door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))

        # Only care once we are "at the entrance" (distance <= 1)
        dist_to_door = self.get_maze_distance(my_pos, door)
        if dist_to_door > 1:
            return True

        risk = self.topology.trap_depth(food_pos)
        required_min_dist = 2 * risk + 1

        visible_ghosts = self._visible_ghost_positions(game_state)
        for gpos in visible_ghosts:
            d = self.get_maze_distance(gpos, door)
            if d <= required_min_dist:
                return False

        return True

    def hunt_food(self, game_state):
        """
        Choose a food target and move toward it.
        - On our side: biased random choice using clusters / risk / beliefs.
        - On enemy side: deterministic nearest-food choice.
        - If chosen food is in a risky pocket, and we are right at its entrance,
          check whether visible ghosts are too close to safely go in.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        if self._should_abort_inside_trap(game_state, my_pos):
            return self.return_home(game_state)
        food_list = self.get_food(game_state).as_list()

        if not food_list or my_pos is None:
            return Directions.STOP

        # Initial target choice
        target = self._pick_food_target(game_state, my_pos, food_list)

        # If that target is in a trap region, and we're at the entrance,
        # check whether it's safe to go in. If not, pick a new (simple) target
        # like on the enemy side: nearest food.
        if target is not None and self.topology.is_in_trap_region(target):
            if not self._safe_to_enter_trap_for_food(game_state, my_pos, target):
                # pick a new target using enemy-side heuristic (nearest)
                target = self._choose_food_enemy_side(game_state, my_pos, food_list)

        if target is None:
            return Directions.STOP

        path = self.pathfinder.find_path(
            game_state,
            my_pos,
            target,
            avoid_enemies=True,
        )

        if path:
            return path[0]

        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP
        return random.choice(legal)

    def return_home(self, game_state):
        """
        Return safely to our side of the map after collecting food.
        Finds the nearest entry tile (across the middle boundary)
        and plans a path back home.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return Directions.STOP

        walls = game_state.get_walls()
        mid_x = walls.width // 2

        # Tiles just on our side of the border
        if self.red:
            home_tiles = [
                (mid_x - 1, y)
                for y in range(walls.height)
                if not walls[mid_x - 1][y]
            ]
        else:
            home_tiles = [
                (mid_x, y)
                for y in range(walls.height)
                if not walls[mid_x][y]
            ]

        # Pick the nearest home tile by maze distance
        if not home_tiles:
            return Directions.STOP

        nearest_home = min(home_tiles, key=lambda t: self.get_maze_distance(my_pos, t))

        # Plan a path home (avoid enemies)
        path = self.pathfinder.find_path(
            game_state,
            my_pos,
            nearest_home,
            avoid_enemies=True
        )

        if path:
            return path[0]

        # Fallback: any legal move
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP
        return random.choice(legal)

    def _should_abort_inside_trap(self, game_state, my_pos):
        """
        Check if we're currently inside a choke corridor and ghosts are too close to the exit.
        Returns True if we should abort current goal and flee.
        """
        if not self.topology.is_in_trap_region(my_pos):
            return False

        pocket = self.topology.pocket_id.get(my_pos)
        if pocket is None:
            return False

        exits = list(self.topology.pocket_exits[pocket])
        if not exits:
            return True  # dead enclosed pocket

        door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))
        pac_to_door = self.get_maze_distance(my_pos, door)

        visible_ghosts = self._visible_ghost_positions(game_state)
        if not visible_ghosts:
            return False

        # Compute shortest ghost distance to that door
        ghost_to_door = min(self.get_maze_distance(g, door) for g in visible_ghosts)

        # If ghost can reach door nearly as fast or faster than we can exit
        if ghost_to_door <= pac_to_door + 1:
            return True

        return False

    def run_away(self, game_state):
        """
        Ghost is nearby on enemy side:
        - If any ghost is 1 tile away → always return home immediately.
        - If 2+ ghosts are close, always return home.
        - Otherwise, probabilistically choose between:
            * returning home
            * going for a new (far-ish) food target.
        A* (pathfinder) handles the actual safe pathing (avoid_enemies=True).
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return Directions.STOP

        radius = 5
        close_ghosts = []
        immediate_danger = False

        for opp in self.get_opponents(game_state):
            st = game_state.get_agent_state(opp)
            pos = st.get_position()
            if pos is not None and not st.is_pacman:
                dist = self.get_maze_distance(my_pos, pos)
                if dist <= 1:
                    # Immediate threat — ghost is next to us!
                    immediate_danger = True
                if dist <= radius:
                    close_ghosts.append(pos)

        # --- Safety overrides ---
        if immediate_danger or len(close_ghosts) >= 2:
            return self.return_home(game_state)

        # --- Probabilistic decision based on carried food ---
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying

        # Probability of going home increases with carried food
        p_home = min(1.0, float(carrying) / max(1, self.CARRY_THRESHOLD))

        if random.random() < p_home:
            return self.return_home(game_state)

        # --- Otherwise, pick a far-ish new food target ---
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return self.return_home(game_state)

        escape_target = self._choose_escape_food_target(game_state, my_pos, food_list)
        if escape_target is None:
            return self.return_home(game_state)

        # Plan path safely
        path = self.pathfinder.find_path(
            game_state,
            my_pos,
            escape_target,
            avoid_enemies=True,
        )

        if path:
            return path[0]

        return self.return_home(game_state)
