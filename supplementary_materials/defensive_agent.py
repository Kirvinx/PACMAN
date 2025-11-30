import random
from contest.capture_agents import CaptureAgent
from contest.game import Directions

# Behavior tree nodes
from supplementary_materials.behavior_tree import (
    Selector, Sequence, Condition, Action
)

# Utilities
from supplementary_materials.exits import TerritoryAnalyzer
from supplementary_materials.topology import MapTopologyAnalyzer
from supplementary_materials.pathfinding_defensive import DefensiveAStarPathfinder
from supplementary_materials.belief_shared import GhostBeliefTracker
from contest.distance_calculator import Distancer
from contest.game import Directions


class BeliefBTDefensiveAgent(CaptureAgent):
    """
    Defensive agent using beliefs + behavior tree.
    
    Modes:
      - Intercept visible invaders on our side
      - Patrol entrances (dual or single defender modes)
    """

    # Shared state across instances
    shared_modes = {}
    shared_targets = {}
    shared_belief_tracker = None
    shared_interceptor = None
    shared_group_assignment = {}
    shared_last_assignment_turn = -1
    shared_distancer = None

    DIR2DELTA = {
        Directions.NORTH: (0, 1),
        Directions.SOUTH: (0, -1),
        Directions.EAST:  (1, 0),
        Directions.WEST:  (-1, 0),
    }

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        
        # Reset shared state
        cls = self.__class__
        cls.shared_modes = {}
        cls.shared_targets = {}
        cls.shared_interceptor = None
        cls.shared_group_assignment = {}
        cls.shared_last_assignment_turn = -1

        # Initialize core components
        self.team_indices = self.get_team(game_state)
        self.mode = "defense"
        self.opponents = self.get_opponents(game_state)
        self.patrol_target = None
        self._avoid_current_threat = None
        
        # Initialize helpers
        self.topology = MapTopologyAnalyzer(game_state.get_walls())
        self.pathfinder = DefensiveAStarPathfinder(self)
        self.territory = TerritoryAnalyzer(game_state, self.red)
        self.home_entries = self._compute_home_entries(game_state)
        walls = game_state.get_walls()
        self.map_width = walls.width
        self.map_height = walls.height
        self.mid_x = self.map_width // 2

        # Initialize shared Distancer
        if cls.shared_distancer is None:
            cls.shared_distancer = Distancer(game_state.data.layout)
            cls.shared_distancer.get_maze_distances()

        self.distancer = cls.shared_distancer

        # Initialize or reuse belief tracker
        if not cls.shared_belief_tracker:
            tracker = GhostBeliefTracker(
                agent=self,
                opponents=self.opponents,
                team_indices=self.team_indices
            )
            tracker.initialize_uniformly(game_state)
            cls.shared_belief_tracker = tracker
        else:
            tracker = cls.shared_belief_tracker
            tracker.debug_agent = self
            tracker.opponents = self.opponents
            tracker.team_indices = self.team_indices
            tracker.initialize_uniformly(game_state)
        
        self.belief_tracker = cls.shared_belief_tracker

        # Build behavior tree
        self.defense_tree = Selector([
            # Intercept visible invaders
            Sequence([
                Condition(lambda a, gs: 
                    a._intruder_inside_or_past_entrance(gs) and 
                    a._am_primary_interceptor(gs)),
                Action(lambda a, gs: a._intercept_invader(gs))
            ]),
            # Patrol based on defender count
            Selector([
                Sequence([
                    Condition(lambda a, gs: a._two_defenders_active(gs)),
                    Action(lambda a, gs: a._patrol_dual(gs))
                ]),
                Action(lambda a, gs: a._patrol_single(gs))
            ])
        ])

        self.behavior_tree = Selector([
            Sequence([
                Condition(lambda a, gs: a.mode == "defense"),
                Action(lambda a, gs: self.defense_tree.execute(a, gs))
            ])
        ])

    def choose_action(self, game_state):
        cls = self.__class__
        cls.shared_modes[self.index] = "defense"

        # Advance motion model (only lowest index agent)
        if self.index == min(self.team_indices):
            self.belief_tracker.elapse_time(game_state)
            if self._two_defenders_active(game_state):
                self._assign_groups_globally(game_state)

        # Observe with sonar
        is_leader = (self.index == min(self.team_indices))
        self.belief_tracker.observe(game_state, self.index, is_leader)

        # Execute behavior tree
        action = self.behavior_tree.execute(self, game_state)
        legal = game_state.get_legal_actions(self.index)
        return action if action in legal else Directions.STOP

    # ------------------------------------------------------------------
    # State checks
    # ------------------------------------------------------------------
    def _entrance_x(self):
        """
        The x-coordinate of the defensive entrance line chosen by TerritoryAnalyzer.
        All entrances lie in exactly this column.
        """
        if self.territory.entrances:
            return self.territory.entrances[0][0]
        # Fallback: use midline border if we somehow have no entrances
        return self.mid_x - 1 if self.red else self.mid_x

    def _is_past_entrance_line(self, pos):
        """
        True if the given position is strictly deeper into our home than the entrance line.
        Being ON the entrance column does NOT count as past.
        """
        if not pos:
            return False

        ex = self._entrance_x()

        if self.red:
            # strictly deeper into home; entrances lie at x == ex
            return pos[0] < ex
        else:
            return pos[0] > ex
        
    def _intruder_inside_or_past_entrance(self, game_state):
        """
        True if any enemy Pacman is inside our territory beyond the entrance line,
        using visibility OR belief peak.
        """
        for opp in self.opponents:
            opp_state = game_state.get_agent_state(opp)
            if not opp_state or not opp_state.is_pacman:
                continue

            # 1) visible
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos and self._is_past_entrance_line(opp_pos):
                return True

            # 2) not visible -> belief peak
            if not opp_pos:
                belief = self.belief_tracker.get_belief(opp)
                if belief:
                    best_pos = max(belief.items(), key=lambda kv: kv[1])[0]
                    if self._is_past_entrance_line(best_pos):
                        return True

        return False

    def _get_scared_timer(self, game_state):
        state = game_state.get_agent_state(self.index)
        if not state:
            return 0
        return getattr(state, "scared_timer", getattr(state, "scaredTimer", 0))

    def _is_scared(self, game_state):
        return self._get_scared_timer(game_state) > 0
    
    def _closest_visible_intruder(self, game_state, max_manhattan=5):
        my_pos = game_state.get_agent_position(self.index)
        if not my_pos:
            return None

        mx, my = my_pos
        best = None
        best_dist = float('inf')

        for opp in self.opponents:
            opp_state = game_state.get_agent_state(opp)
            if not opp_state or not opp_state.is_pacman:
                continue

            opp_pos = game_state.get_agent_position(opp)
            if not opp_pos or not self._is_on_home_side_pos(game_state, opp_pos):
                continue

            dist = abs(mx - opp_pos[0]) + abs(my - opp_pos[1])
            if dist <= max_manhattan and dist < best_dist:
                best = (opp, opp_pos, dist)
                best_dist = dist

        return best
    
    def _two_defenders_active(self, game_state):
        count = sum(1 for idx in self.team_indices 
                   if self.__class__.shared_modes.get(idx) == "defense")
        return count >= 2

    def _any_enemy_visible(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        if not my_pos:
            return False

        mx, my = my_pos
        for opp in self.opponents:
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos and abs(mx - opp_pos[0]) + abs(my - opp_pos[1]) <= 5:
                return True
        return False
    
    def _am_primary_interceptor(self, game_state):
        # TODO: implement proper interceptor selection for multiple defenders
        return True

    def _intruder_inside_or_past_entrance(self, game_state):
        border_x = self._home_border_x(game_state)

        for opp in self.opponents:
            opp_state = game_state.get_agent_state(opp)
            if not opp_state or not opp_state.is_pacman:
                continue

            opp_pos = game_state.get_agent_position(opp)
            if opp_pos:
                ox, _ = opp_pos
                if (self.red and ox < border_x) or (not self.red and ox > border_x):
                    return True
            else:
                # Pacman state but no position = on our side but not visible
                return True
        return False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    
    def _avoid_scared_invader(self, game_state, threat_info):
        """Simplified scared avoidance using maze distance."""
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        if not my_pos:
            return Directions.STOP

        _, threat_pos, _ = threat_info
        dist = self.get_maze_distance(my_pos, threat_pos)

        # Target distance = 2
        if dist == 2 and Directions.STOP in legal:
            return Directions.STOP

        # Evaluate moves
        spawn = game_state.get_initial_agent_position(self.index)
        candidates = []
        for action in legal:
            succ = game_state.generate_successor(self.index, action)
            sp = succ.get_agent_position(self.index)

            if sp == spawn and my_pos != spawn:
                continue

            if sp and self._is_on_home_side_pos(game_state, sp):
                new_dist = self.get_maze_distance(sp, threat_pos)
                candidates.append((action, sp, new_dist))

        if not candidates:
            return Directions.STOP
        
        border_x = self.mid_x - 1 if self.red else self.mid_x

        def border_distance(pos):
            return abs(pos[0] - border_x)

        def dist_to_closest_entrance(pos):
            best = float('inf')
            for ent in self.territory.entrances:
                d = self.distancer.get_distance(pos, ent)
                if d < best:
                    best = d
            return best

        # Move away if too close, toward if too far
        if dist < 2:
            candidates.sort(
                key=lambda c: (
                    -c[2],                          # maximize distance from invader
                    dist_to_closest_entrance(c[1]), # prefer entrance-proximity
                    border_distance(c[1]),          # tie-break near border
                    str(c[0])                       # final tie-break
                )
            )
        elif dist > 2:
            candidates.sort(
                key=lambda c: (
                    c[2],                           # minimize distance to invader
                    dist_to_closest_entrance(c[1]),
                    border_distance(c[1]),
                    str(c[0])
                )
            )
        
        # Avoid traps if possible
        my_depth = self.topology.trap_depth(my_pos)
        inv_depth = self.topology.trap_depth(threat_pos)

        safe = []
        for c in candidates:
            action, sp, new_dist = c
            next_depth = self.topology.trap_depth(sp)

            # Rule 1: Don't go deeper unless invader is deeper
            if my_depth > 0 and next_depth > my_depth and inv_depth <= my_depth:
                continue  # unsafe: entering trap without invader

            safe.append(c)

        # If no safe moves, allow distance-only behavior
        return (safe or candidates)[0][0]

    def _intercept_invader(self, game_state):
        """Intercept invaders using visible positions or belief peaks."""
        if self._is_scared(game_state):
            threat_info = self._closest_visible_intruder(game_state, max_manhattan=5)
            if threat_info:
                return self._avoid_scared_invader(game_state, threat_info)

        my_pos = game_state.get_agent_position(self.index)
        if not my_pos:
            return Directions.STOP

        # --- gather targets exactly as before (visible or belief-based) ---
        intruder_targets = [
            game_state.get_agent_position(opp)
            for opp in self.opponents
            if (opp_state := game_state.get_agent_state(opp)) and opp_state.is_pacman
            and (opp_pos := game_state.get_agent_position(opp))
            and self._is_on_home_side_pos(game_state, opp_pos)
        ]
        if not intruder_targets:
            intruder_targets = []
            for opp in self.opponents:
                opp_state = game_state.get_agent_state(opp)
                if not opp_state or not opp_state.is_pacman:
                    continue
                belief = self.belief_tracker.get_belief(opp)
                if belief:
                    best_pos = max(belief.items(), key=lambda kv: kv[1])[0]
                    if self._is_on_home_side_pos(game_state, best_pos):
                        intruder_targets.append(best_pos)

        if not intruder_targets:
            return Directions.STOP

        target = min(intruder_targets, key=lambda p: self.get_maze_distance(my_pos, p))

        cls = self.__class__
        cls.shared_targets[self.index] = target
        cls.shared_interceptor = self.index

        # NEW: compute path tiles of teammate if I'm the 'follower'
        avoid_tiles = self._predicted_teammate_path_tiles(game_state, target)

        path = self.pathfinder.find_path(
            game_state, my_pos, target,
            avoid_path_tiles=avoid_tiles,
            lambda_path_overlap=100.0,
        )

        legal = game_state.get_legal_actions(self.index)
        if path and path[0] in legal:
            return path[0]

        # Fallback: greedy
        best_action = Directions.STOP
        best_dist = float("inf")
        for action in legal:
            if action != Directions.STOP:
                succ = game_state.generate_successor(self.index, action)
                succ_pos = succ.get_agent_position(self.index)
                if succ_pos:
                    d = self.get_maze_distance(succ_pos, target)
                    if d < best_dist:
                        best_dist, best_action = d, action
        return best_action


    def _patrol_single(self, game_state):
        """Single defender patrol logic."""
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        if not my_pos:
            return Directions.STOP

        if self.__class__.shared_interceptor == self.index:
            self.__class__.shared_interceptor = None

        # Check for visible enemies
        enemy_positions = [
            game_state.get_agent_position(opp)
            for opp in self.opponents
            if game_state.get_agent_position(opp)
        ]

        # MODE A: Enemy visible - use entrance priority
        if enemy_positions:
            self.patrol_target = None
            border_x = self._home_border_x(game_state)
            
            # Get valid successor positions
            action_succ = [
                (action, game_state.generate_successor(self.index, action).get_agent_position(self.index))
                for action in legal
            ]
            
            # Filter to home side
            filtered = [
                (action, pos) for action, pos in action_succ
                if pos and ((self.red and pos[0] <= border_x) or (not self.red and pos[0] >= border_x))
            ]

            if filtered:
                best_action, _ = min(filtered, 
                    key=lambda ap: self.territory.entrance_priority_score(ap[1], enemy_positions))
                return best_action
            return Directions.STOP

        # MODE B: No visible enemy - belief-driven patrol
        entrances = self.territory.entrances
        if not entrances:
            return Directions.STOP

        belief_positions = self._belief_enemy_positions()
        
        if belief_positions:
            d_opp = self.territory._compute_enemy_distances_to_entrances(belief_positions)
            best_idx = min(range(len(entrances)), key=lambda i: d_opp[i])
            patrol_target = entrances[best_idx]
        else:
            patrol_target = entrances[hash(game_state) % len(entrances)]

        path = self.pathfinder.find_path(
            game_state, my_pos, patrol_target
        )


        if path and path[0] in legal:
            return path[0]
        return Directions.STOP

    def _patrol_dual(self, game_state):
        """Dual-defender patrol with entrance group assignment."""
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        if not my_pos:
            return Directions.STOP

        teammate_info = self._get_defensive_teammate_info(game_state)
        if not teammate_info or not teammate_info[1]:
            return self._patrol_single(game_state)

        entrances = self.territory.entrances
        if not entrances:
            return Directions.STOP

        group_a, group_b = getattr(self.territory, "entrance_groups", (None, None))
        if not group_a or not group_b:
            return self._patrol_single(game_state)

        # Get assigned group
        my_group = self.__class__.shared_group_assignment.get(self.index)
        if not my_group:
            return self._patrol_single(game_state)

        my_group_indices = group_a if my_group == "A" else group_b
        assigned_entrances = [entrances[i] for i in my_group_indices]
        #self._debug_draw_entrance_split( assigned_entrances)
        # Pick patrol target
        belief_positions = self._belief_enemy_positions()
        
        if belief_positions:
            d_opp = self.territory._compute_enemy_distances_to_entrances(belief_positions)
            best_idx = min(my_group_indices, key=lambda i: d_opp[i])
            patrol_target = entrances[best_idx]
        else:
            patrol_target = min(assigned_entrances, 
                              key=lambda p: self.get_maze_distance(my_pos, p))

        path = self.pathfinder.find_path(
            game_state, my_pos, patrol_target
        )

        if path and path[0] in legal:
            return path[0]
        return Directions.STOP

    def _return_home(self, game_state):
        """TODO: Navigate to nearest home tile if crossed border."""
        return Directions.STOP

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    
    def _is_on_home_side_pos(self, game_state, pos):
        if not pos:
            return True
        mid_x = game_state.get_walls().width // 2
        return pos[0] < mid_x if self.red else pos[0] >= mid_x

    def _home_border_x(self, game_state):
        mid_x = game_state.get_walls().width // 2
        return (mid_x - 1) if self.red else mid_x

    def _compute_home_entries(self, game_state):
        """Return border tiles that open to enemy side."""
        walls = game_state.get_walls()
        W, H = walls.width, walls.height
        bx = self._home_border_x(game_state)
        across_x = bx + (1 if self.red else -1)

        return [(bx, y) for y in range(H)
                if not walls[bx][y] and 0 <= across_x < W and not walls[across_x][y]]

    def _get_defensive_teammate_info(self, game_state):
        """Get info for another defensive teammate."""
        for idx in self.team_indices:
            if idx != self.index and self.__class__.shared_modes.get(idx) == "defense":
                return (idx, 
                       game_state.get_agent_position(idx),
                       self.__class__.shared_targets.get(idx))
        return None

    def _belief_enemy_positions(self):
        """Get most likely enemy positions from beliefs."""
        positions = []
        for opp in self.opponents:
            belief = self.belief_tracker.get_belief(opp)
            if belief:
                best_pos = max(belief.items(), key=lambda kv: kv[1])[0]
                if best_pos:
                    positions.append(best_pos)
        return positions

    def _assign_groups_globally(self, game_state):
        """Assign defenders to entrance groups (called once per turn)."""
        group_a, group_b = getattr(self.territory, "entrance_groups", (None, None))
        if not group_a or not group_b:
            return

        # Get active defenders
        defenders = [
            (idx, game_state.get_agent_position(idx))
            for idx in self.team_indices
            if self.__class__.shared_modes.get(idx) == "defense"
            and game_state.get_agent_position(idx)
        ]

        if len(defenders) < 2:
            return

        defenders.sort(key=lambda t: t[0])
        (idx1, pos1), (idx2, pos2) = defenders[:2]

        # Distance to group helper
        def dist_to_group(pos, group):
            entrances = self.territory.entrances
            return min(self.get_maze_distance(pos, entrances[gi]) for gi in group)

        # Find optimal assignment
        cost1 = dist_to_group(pos1, group_a) + dist_to_group(pos2, group_b)
        cost2 = dist_to_group(pos1, group_b) + dist_to_group(pos2, group_a)

        assignment = {idx1: "A", idx2: "B"} if cost1 <= cost2 else {idx1: "B", idx2: "A"}
        self.__class__.shared_group_assignment = assignment

    def _pick_patrol_target(self, game_state):
        """Choose entrance to patrol (unused but kept for future implementation)."""
        entrances = self.territory.entrances
        if not entrances:
            return None

        enemy_positions = self.territory.get_most_likely_enemy_positions(
            self.belief_tracker, self.opponents)

        if not enemy_positions:
            return random.choice(entrances)

        d_opp = self.territory._compute_enemy_distances_to_entrances(enemy_positions)
        best_idx = min(range(len(entrances)), key=lambda i: d_opp[i])
        
        return entrances[best_idx] if random.random() < 0.7 else random.choice(entrances)

    def _debug_draw_entrance_split(self, assigned_entrances):
        """Debug visualization for entrance assignments."""
        if not hasattr(self, "debug_draw") or self.index != min(self.team_indices):
            return

        all_entrances = self.territory.entrances
        assigned_set = set(assigned_entrances)
        mine = list(assigned_set)
        theirs = [e for e in all_entrances if e not in assigned_set]

        if hasattr(self, "debug_clear"):
            self.debug_clear()

        if mine:
            self.debug_draw(mine, [0.0, 0.0, 1.0], clear=False)
        if theirs:
            self.debug_draw(theirs, [0.0, 1.0, 0.0], clear=False)

    def _path_to_positions(self, start_pos, actions):
        """Convert a list of actions into the sequence of grid positions visited."""
        x, y = start_pos
        tiles = []
        for a in actions:
            if a not in self.DIR2DELTA:
                continue
            dx, dy = self.DIR2DELTA[a]
            x += dx
            y += dy
            tiles.append((x, y))
        return tiles

    def _predicted_teammate_path_tiles(self, game_state, target):
        """
        If there is exactly one other defender, and I'm the 'follower'
        (i.e., farther from the intruder target), approximate the path
        that teammate would take to the same target and return its tiles
        as a set for overlap-avoid penalization.
        """
        # Only use this when we actually have 2 defenders
        if not self._two_defenders_active(game_state):
            return None

        teammate_info = self._get_defensive_teammate_info(game_state)
        if not teammate_info:
            return None

        mate_idx, mate_pos, mate_target = teammate_info
        my_pos = game_state.get_agent_position(self.index)

        if not my_pos or not mate_pos or not target:
            return None

        # --- DISTANCE-BASED LEADER SELECTION ---
        # Compute maze distances to the same intruder target
        d_self = self.get_maze_distance(my_pos, target)
        d_mate = self.get_maze_distance(mate_pos, target)

        # Leader = closer defender; tie-break by index (lower index = leader)
        i_am_leader = (
            d_self < d_mate or
            (d_self == d_mate and self.index < mate_idx)
        )

        if i_am_leader:
            # I'm the closer one → I take the direct path (no overlap penalty)
            return None

        # I'm the follower (farther from intruder). I should path around
        # the teammate's "ideal" path.
        mate_path = self.pathfinder.find_path(
            game_state,
            start=mate_pos,
            goal=target
        )

        if not mate_path:
            return None

        tiles = self._path_to_positions(mate_pos, mate_path)

        # Typically you don't want to penalize the final target tile,
        # so both defenders can converge on the intruder.
        if tiles:
            tiles = tiles[:-1]

        return set(tiles)



# TODO: COULD WE MAKE SOME METHOD WHERE WE CHASE IN SUCH WAY TO TRY TO KEEP OURSELF CLOSE TO BORDER?