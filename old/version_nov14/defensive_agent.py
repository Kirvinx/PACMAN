# contest/agents/team_name_1/defense/defensive_agent_template.py

import random

from contest.capture_agents import CaptureAgent
from contest.game import Directions

# Behavior tree nodes (same ones you use on offense)
from contest.agents.team_name_1.baseline.behavior_tree import (
    Selector,
    Sequence,
    Condition,
    Action,
)

# Optional utilities you’ll likely want later
from contest.agents.team_name_1.beliefline.exits import TerritoryAnalyzer
from contest.agents.team_name_1.beliefline.topology import MapTopologyAnalyzer
from contest.agents.team_name_1.beliefline.pathfinding_defensive import DefensiveAStarPathfinder
from contest.agents.team_name_1.beliefline.belief_shared import GhostBeliefTracker


class BeliefBTDefensiveAgent(CaptureAgent):
    """
    Defensive agent using beliefs + behavior tree.

    Modes:
      - Intercept visible invaders on our side.
      - Patrol entrances, with two styles:
          * If THIS defender sees an enemy anywhere: local entrance-ETA heuristic.
          * If no enemy visible to us: A* patrol between entrances with ally-avoidance off.
    """

    # ---- Shared state across instances of *this* class ----
    shared_modes = {}       # agent_index -> "defense"
    shared_targets = {}     # agent_index -> (x, y) current intent/target (optional)
    shared_belief_tracker = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # Team info & static helpers
        self.team_indices = self.get_team(game_state)
        self.mode = "defense"
        self.topology = MapTopologyAnalyzer(game_state.get_walls())
        self.pathfinder = DefensiveAStarPathfinder(self)
        self.territory = TerritoryAnalyzer(game_state, self.red)
        self.opponents = self.get_opponents(game_state)
        if BeliefBTDefensiveAgent.shared_belief_tracker is None:
            # First defender creates & initializes shared tracker
            BeliefBTDefensiveAgent.shared_belief_tracker = GhostBeliefTracker(
                agent=self,                # used only for team color + debug
                opponents=self.opponents,
                team_indices=self.team_indices
            )
            BeliefBTDefensiveAgent.shared_belief_tracker.initialize_uniformly(game_state)
        self.belief_tracker = BeliefBTDefensiveAgent.shared_belief_tracker
        # Patrol state (target entrance / patrol point)
        self.patrol_target = None

        # Precompute (optional) home-side geometry
        self.home_entries = self._compute_home_entries(game_state)

        # ---- DEFENSE SUBTREE ----
        self.defense_tree = Selector([
            # 1) If a visible invader is on our side, intercept
            Sequence([
                Condition(lambda agent, gs: agent._visible_invader_on_our_side(gs)),
                Action(lambda agent, gs: agent._intercept_invader(gs)),
            ]),
            # 2) Otherwise patrol: dual vs single
            Selector([
                Sequence([
                    Condition(lambda agent, gs: agent._two_defenders_active(gs)),
                    Action(lambda agent, gs: agent._patrol_dual(gs)),
                ]),
                Action(lambda agent, gs: agent._patrol_single(gs)),
            ]),
        ])

        # ---- TOP-LEVEL BT ----
        self.behavior_tree = Selector([
            Sequence([
                Condition(lambda agent, gs: agent.mode == "defense"),
                Action(lambda agent, gs: agent.defense_tree.execute(agent, gs)),
            ]),
        ])

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def choose_action(self, game_state):
        # Advertise our mode (coordination hook)
        BeliefBTDefensiveAgent.shared_modes[self.index] = "defense"

        # 1) Advance motion model ONLY once per team turn
        #    Use the lower-index teammate as the "time advancer"
        if self.index == min(self.team_indices):
            self.belief_tracker.elapse_time(game_state)

        # 2) EVERY defender contributes its sonar for this step
        leader_index = min(self.team_indices)
        is_leader = (self.index == leader_index)

        self.belief_tracker.observe(
            game_state,
            observing_index=self.index,
            is_leader=is_leader
        )

        # 3) Run behavior tree as before
        action = self.behavior_tree.execute(self, game_state)

        legal = game_state.get_legal_actions(self.index)
        return action if action in legal else (Directions.STOP if legal else Directions.STOP)

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def _two_defenders_active(self, game_state):
        """
        Returns True if at least two of our team agents use this class
        and are in defensive mode.

        TODO: Implement real check. For now, return False to default to single mode.
        """
        return False

    def _visible_invader_on_our_side(self, game_state):
        """
        Returns True if a visible opponent (as Pacman) is on our side.

        TODO: Inspect opponent agent states and positions.
        """
        return False

    def _any_enemy_visible(self, game_state):
        """
        Returns True ONLY if THIS defender can see an enemy,
        according to capture visibility rules (Manhattan distance <= 5).
        """
        my_pos = game_state.get_agent_position(self.index)
        if my_pos is None:
            return False

        mx, my = my_pos

        for opp in self.opponents:
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos is None:
                continue
            ox, oy = opp_pos
            if abs(mx - ox) + abs(my - oy) <= 5:
                return True

        return False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _intercept_invader(self, game_state):
        """
        TODO: Pick nearest visible invader on our side and A* towards it
              (optionally with ally-repulsion and danger map).
        """
        return Directions.STOP

    # ---- PATROL HELPERS ----

    def _pick_patrol_target(self, game_state):
        """
        Choose an entrance tile to patrol toward.

        - If we have belief-based likely enemy positions, bias toward the entrance
          that is closest (in TerritoryAnalyzer's entrance-distance metric).
        - Otherwise, pick a random entrance.
        """
        entrances = self.territory.entrances
        if not entrances:
            return None

        enemy_positions = self.territory.get_most_likely_enemy_positions(
            self.belief_tracker,
            self.opponents,
        )

        # No useful enemy info -> pure random
        if not enemy_positions:
            return random.choice(entrances)

        # Use existing distance machinery to find which entrance is "most threatened"
        d_opp = self.territory._compute_enemy_distances_to_entrances(enemy_positions)
        best_idx = min(range(len(entrances)), key=lambda i: d_opp[i])

        # Slight randomness: 70% choose the best, 30% choose any entrance
        if random.random() < 0.7:
            return entrances[best_idx]
        else:
            return random.choice(entrances)

    def _patrol_single(self, game_state):
        """
        Defensive behavior when no visible invaders are on our side.

        MODE A: enemy visible
            -> Use lexicographic entrance defense on legal successor tiles.

        MODE B: no visible enemy
            -> Use persistent A* patrol path with ally-avoidance OFF.
            The agent continues following the existing patrol path
            until it finishes, then picks a new entrance target.
        """
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        if my_pos is None:
            return Directions.STOP

        # ----------------------------------------------------------
        # Check for visible enemies
        # ----------------------------------------------------------
        enemy_positions = []
        for opp in self.opponents:
            pos = game_state.get_agent_position(opp)
            if pos is not None:
                enemy_positions.append(pos)

        # ----------------------------------------------------------
        # MODE A — enemy visible
        # ----------------------------------------------------------
        if enemy_positions:
            # When we see an enemy, discard any ongoing patrol path
            self.current_patrol_path = None
            self.patrol_target = None

            action_succ = []
            for action in legal:
                successor = game_state.generate_successor(self.index, action)
                new_pos = successor.get_agent_position(self.index)
                if new_pos is not None:
                    action_succ.append((action, new_pos))

            border_x = self._home_border_x(game_state)
            filtered = []
            for action, pos in action_succ:
                x, y = pos
                if (self.red and x <= border_x) or ((not self.red) and x >= border_x):
                    filtered.append((action, pos))

            if not filtered:
                return Directions.STOP

            best_action, _ = min(
                filtered,
                key=lambda ap: self.territory.entrance_priority_score(ap[1], enemy_positions)
            )
            return best_action

        # ----------------------------------------------------------
        # MODE B — no visible enemy → belief-driven patrol (no persistence)
        # ----------------------------------------------------------
        entrances = self.territory.entrances
        if not entrances:
            return Directions.STOP

        # Get likely enemy positions from beliefs
        belief_enemy_positions = self._belief_enemy_positions()

        if belief_enemy_positions:
            # Use entrance distance machinery to find most threatened entrance
            d_opp = self.territory._compute_enemy_distances_to_entrances(
                belief_enemy_positions
            )
            best_idx = min(range(len(entrances)), key=lambda i: d_opp[i])
            patrol_target = entrances[best_idx]
        else:
            # No useful belief info → simple fallback
            patrol_target = entrances[hash(game_state) % len(entrances)]

        path = self.pathfinder.find_path(
            game_state,
            start=my_pos,
            goal=patrol_target,
            avoid_allies=False,
            lambda_ally=0.0,
            repulsion_radius=0,
            decay_alpha=0.0,
        )

        if not path:
            return Directions.STOP

        first_action = path[0]
        if first_action in legal:
            return first_action

        return Directions.STOP



    def _patrol_dual(self, game_state):
        """
        TODO: Split home entries with teammate (e.g., Voronoi by distance) and
              repel via DefensiveAStarPathfinder to keep separation.
        """
        return Directions.STOP

    def _return_home(self, game_state):
        """
        TODO: Navigate to nearest home tile if we cross the border by accident.
        """
        return Directions.STOP

    # ------------------------------------------------------------------
    # Geometry & map helpers
    # ------------------------------------------------------------------

    def _is_on_home_side_pos(self, game_state, pos):
        if pos is None:
            return True
        mid_x = game_state.get_walls().width // 2
        return pos[0] < mid_x if self.red else pos[0] >= mid_x

    def _home_border_x(self, game_state):
        mid_x = game_state.get_walls().width // 2
        return mid_x - 1 if self.red else mid_x

    def _compute_home_entries(self, game_state):
        """
        Return tiles on our border column that open to enemy side.
        Kept simple; safe to use as-is or replace later.
        """
        walls = game_state.get_walls()
        W, H = walls.width, walls.height
        bx = self._home_border_x(game_state)
        step = +1 if self.red else -1
        across_x = bx + step

        entries = []
        for y in range(H):
            if walls[bx][y]:
                continue
            if 0 <= across_x < W and not walls[across_x][y]:
                entries.append((bx, y))
        return entries

    def _get_defensive_teammate_info(self, game_state):
        """
        Optional helper: return (idx, pos, target) of another defender.

        TODO: Fill if you want dual-mode coordination data.
        """
        return None

    def _belief_enemy_positions(self):
        """
        Turn belief distributions into a list of 'most likely' enemy positions.
        One guess per opponent.
        """
        positions = []

        for opp in self.opponents:
            belief = self.belief_tracker.get_belief(opp)
            if not belief:
                continue

            # argmax over belief
            best_pos, best_p = None, 0.0
            for pos, p in belief.items():
                if p > best_p:
                    best_pos, best_p = pos, p

            if best_pos is not None:
                positions.append(best_pos)

        return positions