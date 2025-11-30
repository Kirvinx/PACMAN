from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import manhattan_distance
from supplementary_materials.pathfinding_belief_opt import AStarPathfinder
from supplementary_materials.pathfinding_defensive import DefensiveAStarPathfinder
from supplementary_materials.belief_shared import GhostBeliefTracker
from supplementary_materials.behavior_tree import (
    Selector, Sequence, Condition, Action
)
from supplementary_materials.topology import MapTopologyAnalyzer
import random
from collections import deque


class BeliefBTOffensiveAgent(CaptureAgent):
    """
    Optimized offensive agent using:
      - GhostBeliefTracker (probabilistic ghost locations)
      - Belief-based A* pathfinding
      - Behavior Tree for high-level decisions
    """

    CARRY_THRESHOLD = 10
    ESCAPE_PELLET_RISK_FACTOR = 0.9
    
    # Shared class variables
    shared_targets = {}
    shared_modes = {}
    shared_belief_tracker = None

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # Initialize instance variables
        self.recent_positions = deque(maxlen=8)
        self.stuck_counter = 0
        self.team_indices = self.get_team(game_state)
        self.topology = MapTopologyAnalyzer(game_state.get_walls())
        self.pathfinder = AStarPathfinder(self)
        self.defensive_pathfinder = DefensiveAStarPathfinder(self)
        self.opponents = self.get_opponents(game_state)
        self.mode = "offense"
        
        self.last_intent = None
        self.deadlock_target = None
        self.deadlock_commit = 0
        self.in_trap_escape = False
        self.trap_escape_path = None
        self.deadlock = 0
        
        # Cache frequently used values
        walls = game_state.get_walls()
        self.map_width = walls.width
        self.map_height = walls.height
        self.mid_x = self.map_width // 2

        
        
        # Initialize shared belief tracker with proper reinitialization
        if self.__class__.shared_belief_tracker is None:
            tracker = GhostBeliefTracker(
                agent=self,
                opponents=self.opponents,
                team_indices=self.team_indices,
            )
            tracker.initialize_uniformly(game_state)
            self.__class__.shared_belief_tracker = tracker
        else:
            # Reinitialize for subsequent agents or new games
            tracker = self.__class__.shared_belief_tracker
            tracker.debug_agent = self
            tracker.opponents = self.opponents
            tracker.team_indices = self.team_indices
            tracker.initialize_uniformly(game_state)

        self.belief_tracker = self.__class__.shared_belief_tracker
        self.__class__.shared_targets = {}
        self.__class__.shared_modes = {}
        
        # Build behavior tree using lambdas to reduce boilerplate
        self.offense_tree = self._build_offense_tree()
        self.behavior_tree = Selector([
            Sequence([
                Condition(lambda a, gs: a.mode == "offense"),
                Action(lambda a, gs: a.offense_tree.execute(a, gs)),
            ])
        ])

    def _build_offense_tree(self):
        """Build offense behavior tree with optimized conditions."""
        return Selector([
            # Deadlock target pursuit
            Sequence([
                Condition(lambda a, gs: a.deadlock_target is not None),
                Action(lambda a, gs: a._pursue_deadlock_target(gs)),
            ]),
            # Deadlock resolution
            Sequence([
                Condition(lambda a, gs: a.stuck_counter >= 3),
                Action(lambda a, gs: a.break_deadlock(gs)),
            ]),
            # Emergency retreat
            Sequence([
                Condition(lambda a, gs: a._should_retreat(gs)),
                Action(lambda a, gs: a.run_away(gs)),
            ]),
            # Return home if carrying threshold met
            Sequence([
                Condition(lambda a, gs: 
                    gs.get_agent_state(a.index).num_carrying >= a.CARRY_THRESHOLD),
                Action(lambda a, gs: a.return_home(gs)),
            ]),
            # Food hunting
            Selector([
                Sequence([
                    Condition(lambda a, gs: a._two_offensive_agents(gs)),
                    Action(lambda a, gs: a.hunt_food_dual(gs)),
                ]),
                Action(lambda a, gs: a.hunt_food_single(gs)),
            ]),
        ])

    def choose_action(self, game_state):
        """Main action selection with optimized belief updates."""
        # Shared belief update
        leader_index = min(self.team_indices)
        is_leader = (self.index == leader_index)

        if is_leader:
            self.belief_tracker.elapse_time(game_state)

        self.belief_tracker.observe(game_state, self.index, is_leader)

        # Handle trap escape if active
        if self.in_trap_escape:
            action = self._continue_trap_escape(game_state)
            if action is not None:
                legal = game_state.get_legal_actions(self.index)
                if action in legal:
                    return action
                else:
                    # Abort trap escape if producing illegal actions
                    self.in_trap_escape = False
                    self.trap_escape_path = None
                    return self._safe_fallback_toward_home(game_state)

        # Update position tracking for deadlock detection
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        if my_pos is not None:
            self.recent_positions.append(my_pos)

        if self._is_stuck_deadlock():
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # Update shared mode
        self.__class__.shared_modes[self.index] = self.mode

        # Execute behavior tree
        action = self.behavior_tree.execute(self, game_state)
        
        legal = game_state.get_legal_actions(self.index)
        return action if action in legal else self._safe_fallback_toward_home(game_state)

    def is_on_enemy_side(self, game_state):
        """Check if on opponent's half - optimized with cached mid_x."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return False
        return my_pos[0] >= self.mid_x if self.red else my_pos[0] < self.mid_x

    def is_power_active(self, game_state):
        """Check if any enemy ghost is scared."""
        return any(game_state.get_agent_state(opp).scared_timer > 3
                  for opp in self.opponents)

    def _is_dangerous_ghost(self, opp_state):
        """Check if enemy ghost is dangerous."""
        return not opp_state.is_pacman and opp_state.scared_timer <= 3

    def _two_offensive_agents(self, game_state):
        """Check if at least two team agents are offensive."""
        offensive_count = sum(1 for idx in self.team_indices
                            if self.__class__.shared_modes.get(idx, "offense") == "offense")
        return offensive_count >= 2

    def _should_retreat(self, game_state):
        """Optimized retreat condition check."""
        return (self.is_on_enemy_side(game_state) 
                and not self.is_power_active(game_state)
                and self.ghost_nearby(game_state, radius=5))

    def ghost_nearby(self, game_state, radius=5):
        """Optimized ghost proximity check with early exit."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return False

        for opp in self.opponents:
            opp_state = game_state.get_agent_state(opp)
            if not self._is_dangerous_ghost(opp_state):
                continue
                
            opp_pos = opp_state.get_position()
            if opp_pos and self.get_maze_distance(my_pos, opp_pos) <= radius:
                return True
        return False

    def break_deadlock(self, game_state):
        """Handle deadlock with optimized side detection."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP

        self.stuck_counter = 0
        self.recent_positions.clear()

        if self.is_on_enemy_side(game_state):
            return self._break_deadlock_enemy_side(game_state, my_pos)
        return self._break_deadlock_home_side(game_state, my_pos)

    def _break_deadlock_home_side(self, game_state, my_pos):
        """Home-side deadlock breaker matching enemy-side logic."""
        # Generate list of border tiles (last defensive column on our side)
        walls = game_state.get_walls()
        width, height = walls.width, walls.height

        if self.red:
            border_x = self.mid_x - 1  # Last column on red side
        else:
            border_x = self.mid_x  # First column on blue side

        # Get all non-wall tiles at the border
        border_tiles = [(border_x, y) for y in range(height) if not walls[border_x][y]]

        if not border_tiles:
            return self._safe_fallback_toward_home(game_state)

        # Avoid picking a tile extremely close
        MIN_DIST = 2
        far_tiles = [t for t in border_tiles
                    if self.get_maze_distance(my_pos, t) >= MIN_DIST]
        candidates = far_tiles if far_tiles else border_tiles
        # Pick random tile and set target
        target = random.choice(candidates)
        
        self.deadlock_target = target
        self.deadlock_commit = 10 #TODO: OKAY?

        # Find initial path
        path = self.defensive_pathfinder.find_path(
            game_state,
            start=my_pos,
            goal=target,
            avoid_allies=False,
            return_cost=False
        )

        if path:
            return path[0]  # ← Just return, don't decrement here!

        # Path failed → clean reset and fallback
        self.deadlock_target = None
        self.deadlock_commit = 0
        return self._safe_fallback_toward_home(game_state)

    def _break_deadlock_enemy_side(self, game_state, my_pos):
        """Handle deadlock on enemy side with optimized logic."""
        if self.last_intent != "run_home" and self.deadlock == 0:
            self.deadlock = 1
            print("hmm")
            return self._escape_via_best_option(game_state)

        target = self._pick_deep_enemy_target(game_state, my_pos)
        if not target:
            return self._safe_fallback_toward_home(game_state)

        self.deadlock_target = target
        self.deadlock_commit = 6

        path = self.pathfinder.find_path(game_state, my_pos, target, avoid_enemies=True)
        if path:
            return path[0]

        self.deadlock_target = None
        self.deadlock_commit = 0
        return self._safe_fallback_toward_home(game_state)
    
    def _is_stuck_deadlock(self):
        """Detect movement deadlocks - optimized with single pass."""
        pos_list = list(self.recent_positions)
        if len(pos_list) < self.recent_positions.maxlen:
            return False

        # Check if stuck in same position for 4 moves
        a, b, c, d = pos_list[-4:]
        if a == b == c == d:
            return True
        
        # Check A-B-A-B oscillation pattern
        return a == c and b == d and a != b

    def _pursue_deadlock_target(self, game_state):
        """Pursue deadlock target with optimized cleanup."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos or not self.deadlock_target:
            self._clear_deadlock_state()
            return Directions.STOP

        if self.deadlock_commit <= 0:
            self._clear_deadlock_state()
            return Directions.STOP

        # Choose pathfinder based on which side we're on
        if self.is_on_enemy_side(game_state):
            path = self.pathfinder.find_path(
                game_state, my_pos, self.deadlock_target, avoid_enemies=True
            )
        else:
            # Use defensive pathfinder for home side
            path = self.defensive_pathfinder.find_path(
                game_state,
                start=my_pos,
                goal=self.deadlock_target,
                avoid_allies=False,
                return_cost=False
            )

        if path:
            self.deadlock_commit -= 1
            if my_pos == self.deadlock_target or \
            self.get_maze_distance(my_pos, self.deadlock_target) <= 1:
                self._clear_deadlock_state()
            return path[0]

        self._clear_deadlock_state()
        return self._safe_fallback_toward_home(game_state)

    def _clear_deadlock_state(self):
        """Helper to clear deadlock state."""
        self.deadlock_target = None
        self.deadlock_commit = 0
        self.deadlock = 0

    def _pick_deep_enemy_target(self, game_state, my_pos):
        """Optimized deep enemy target selection."""
        # Determine enemy side boundaries
        enemy_min_x = self.mid_x if self.red else 0
        enemy_max_x = self.map_width - 1 if self.red else self.mid_x - 1

        # Get all legal enemy tiles efficiently
        walls = game_state.get_walls()
        enemy_legal = [(x, y) for x in range(enemy_min_x, enemy_max_x + 1)
                      for y in range(self.map_height)
                      if not walls[x][y]]

        if not enemy_legal:
            return None

        # Check for deeper food
        food_list = self.get_food(game_state).as_list()
        enemy_food = [f for f in food_list if enemy_min_x <= f[0] <= enemy_max_x]

        if enemy_food:
            depth_threshold = my_pos[0] + 3 if self.red else my_pos[0] - 3
            deeper_food = [f for f in enemy_food 
                          if (f[0] >= depth_threshold if self.red else f[0] <= depth_threshold)]
            
            if deeper_food:
                return random.choice(deeper_food)

        return random.choice(enemy_legal)

    def _get_visible_ghost_positions(self, game_state):
        """Get visible dangerous ghost positions."""
        positions = []
        for opp in self.opponents:
            st = game_state.get_agent_state(opp)
            pos = st.get_position()
            if pos is not None and self._is_dangerous_ghost(st):
                positions.append(pos)
        return positions

    def get_believed_ghost_positions(self, game_state):
        """Get believed ghost positions with optimized lookup."""
        positions = []
        for opp in self.opponents:
            belief_dist = self.belief_tracker.get_belief(opp)
            if belief_dist:
                positions.append(max(belief_dist.items(), key=lambda kv: kv[1])[0])
            else:
                pos = game_state.get_agent_state(opp).get_position()
                if pos:
                    positions.append(pos)
        return positions

    def _get_home_tiles(self, game_state):
        """Get list of tiles on our side of the border."""
        walls = game_state.get_walls()
        x = self.mid_x - 1 if self.red else self.mid_x
        return [(x, y) for y in range(self.map_height) if not walls[x][y]]

    def _pick_food_target_single(self, game_state, my_pos, food_list):
        """Optimized single agent food targeting."""
        if not food_list:
            return self.return_home(game_state)

        if self.is_on_enemy_side(game_state):
            return min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))

        # Score-based selection for home side
        believed_ghosts = self.get_believed_ghost_positions(game_state)
        
        def score_food(food_pos):
            cluster_size = sum(1 for f in food_list 
                            if manhattan_distance(food_pos, f) <= 3) - 1
            risk_depth = self.topology.trap_depth(food_pos)
            ghost_dist = (min(self.get_maze_distance(food_pos, g) for g in believed_ghosts) 
                        if believed_ghosts else 10)
            return 1.0 * cluster_size + 3.0 * ghost_dist - 2.0 * risk_depth
        
        # Score all food
        candidates = [(score_food(f), f) for f in food_list]
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Weighted random selection using exponential weights
        scores = [score for score, _ in candidates]
        min_score = min(scores)

        e = 2.71828
        weights = [e ** (0.5 * (score - min_score)) for score in scores]
        total_weight = sum(weights)

        # Normalize to probabilities
        probabilities = [w / total_weight for w in weights]

        # Weighted random choice
        rand_val = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand_val <= cumulative:
                return candidates[i][1]

        # Fallback (shouldn't reach here)
        return candidates[0][1]

    def _pick_food_target_dual(self, game_state, my_pos, food_list):
        """Optimized dual offense targeting."""
        if not food_list:
            return self.return_home(game_state)

        mate_info = self._get_offensive_teammate_info(game_state)
        if not mate_info:
            return self._pick_food_target_single(game_state, my_pos, food_list)

        _, mate_pos, mate_target = mate_info
        ref_pos = mate_target or mate_pos
        MIN_SEP = 3

        non_overlap = [f for f in food_list
                      if self.get_maze_distance(f, ref_pos) >= MIN_SEP]

        if not non_overlap:
            return None

        def cost(f):
            d_self = self.get_maze_distance(my_pos, f)
            d_mate = self.get_maze_distance(ref_pos, f)
            return d_self - 0.6 * d_mate

        return min(non_overlap, key=cost)

    def _safe_to_enter_trap(self, game_state, my_pos, food_pos):
        """Optimized trap safety check."""
        if not self.topology.is_in_trap_region(food_pos):
            return True

        pocket = self.topology.pocket_id.get(food_pos)
        if not pocket:
            return True

        exits = list(self.topology.pocket_exits[pocket])
        if not exits:
            return False

        door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))

        if self.get_maze_distance(my_pos, door) > 1:
            return True

        risk = self.topology.trap_depth(food_pos)
        required_dist = 2 * risk + 1

        visible_ghosts = self._get_visible_ghost_positions(game_state)
        return all(self.get_maze_distance(g, door) > required_dist for g in visible_ghosts)

    def _should_abort_trap(self, game_state, my_pos):
        """Optimized trap abort check."""
        depth = self.topology.trap_depth(my_pos)
        if depth <= 0:
            return False

        pocket = self.topology.pocket_id.get(my_pos)
        if not pocket:
            return False

        exits = list(self.topology.pocket_exits[pocket])
        door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))
        pac_to_door = self.get_maze_distance(my_pos, door)

        visible_ghosts = self._get_visible_ghost_positions(game_state)
        if not visible_ghosts:
            return False

        ghost_to_door = min(self.get_maze_distance(g, door) for g in visible_ghosts)
        return ghost_to_door <= pac_to_door + 5

    def hunt_food_single(self, game_state):
        """Single agent food hunting."""
        return self._hunt_food_impl(game_state, dual=False)

    def hunt_food_dual(self, game_state):
        """Dual agent food hunting."""
        return self._hunt_food_impl(game_state, dual=True)

    def _hunt_food_impl(self, game_state, dual: bool):
        """Unified food hunting implementation."""
        self.last_intent = "food"
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP

        if self._should_abort_trap(game_state, my_pos):
            return self._begin_trap_escape(game_state)

        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return self.return_home(game_state)

        # Select target based on mode
        target = (self._pick_food_target_dual(game_state, my_pos, food_list) if dual
                 else self._pick_food_target_single(game_state, my_pos, food_list))

        if not target:
            return self.return_home(game_state) if dual and self.is_on_enemy_side(game_state) else Directions.STOP

        # Store target for coordination
        self.__class__.shared_targets[self.index] = target

        # Safety check for trap targets
        if (self.topology.is_in_trap_region(target) 
            and not self._safe_to_enter_trap(game_state, my_pos, target)):
            target = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))

        if not target:
            return Directions.STOP

        path = self.pathfinder.find_path(game_state, my_pos, target, avoid_enemies=True)
        return path[0] if path else self._safe_fallback_toward_home(game_state)

    def return_home(self, game_state):
        """Optimized return home logic."""
        self.last_intent = "run_home"
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP

        home_tiles = self._get_home_tiles(game_state)
        if not home_tiles:
            return Directions.STOP

        nearest_home = min(home_tiles, key=lambda t: self.get_maze_distance(my_pos, t))
        path = self.pathfinder.find_path(game_state, my_pos, nearest_home, avoid_enemies=True)

        return path[0] if path else self._safe_fallback_toward_home(game_state)

    def run_away(self, game_state):
        """Optimized escape logic."""
        self.last_intent = "escape"
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP

        # Check danger level
        close_ghosts = []
        immediate_danger = False

        for opp in self.opponents:
            st = game_state.get_agent_state(opp)
            pos = st.get_position()
            if pos and not st.is_pacman:
                dist = self.get_maze_distance(my_pos, pos)
                if dist <= 1:
                    immediate_danger = True
                if dist <= 5:
                    close_ghosts.append(pos)

        # Decide escape strategy
        my_state = game_state.get_agent_state(self.index)
        p_home = min(1.0, float(my_state.num_carrying) / max(1, self.CARRY_THRESHOLD))

        if immediate_danger or len(close_ghosts) >= 2 or random.random() < p_home:
            return self._escape_via_best_option(game_state)

        # Try random food escape
        food_list = self.get_food(game_state).as_list()
        if food_list:
            weighted = [(f, max(1, self.get_maze_distance(my_pos, f))**2) for f in food_list]
            total_w = sum(w for _, w in weighted)

            if total_w > 0:
                r = random.uniform(0, total_w)
                acc = 0.0
                for f, w in weighted:
                    acc += w
                    if r <= acc:
                        escape_target = f
                        break
                else:
                    escape_target = weighted[-1][0]

                path = self.pathfinder.find_path(
                    game_state, my_pos, escape_target, avoid_enemies=True
                )
                if path:
                    return path[0]

        return self._escape_via_best_option(game_state)

    def _escape_via_best_option(self, game_state):
        """Optimized escape option selection."""
        self.last_intent = "run_home"
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP

        home_tiles = self._get_home_tiles(game_state)
        if not home_tiles:
            return Directions.STOP

        nearest_home = min(home_tiles, key=lambda t: self.get_maze_distance(my_pos, t))

        if self.is_power_active(game_state):
            path, _ = self.pathfinder.find_path(
                game_state, my_pos, nearest_home, avoid_enemies=False, return_cost=True
            )
            if path:
                return path[0]
        else:
            pellets = self.get_capsules(game_state)

            if pellets:
                nearest_pellet = min(pellets, key=lambda c: self.get_maze_distance(my_pos, c))

                home_path, home_cost = self.pathfinder.find_path(
                    game_state, my_pos, nearest_home, avoid_enemies=True, return_cost=True
                )
                pellet_path, pellet_cost = self.pathfinder.find_path(
                    game_state, my_pos, nearest_pellet, avoid_enemies=True, return_cost=True
                )

                if pellet_path and (not home_path or 
                    home_cost > self.ESCAPE_PELLET_RISK_FACTOR * pellet_cost):
                    return pellet_path[0]
                elif home_path:
                    return home_path[0]
            else:
                path, _ = self.pathfinder.find_path(
                    game_state, my_pos, nearest_home, avoid_enemies=True, return_cost=True
                )
                if path:
                    return path[0]

        return self._safe_fallback_toward_home(game_state)

    def _get_offensive_teammate_info(self, game_state):
        """Optimized teammate info retrieval."""
        for idx in self.team_indices:
            if idx == self.index:
                continue

            if self.__class__.shared_modes.get(idx) != "offense":
                continue

            st = game_state.get_agent_state(idx)
            if st is not None:
                pos = st.get_position()
                if pos is not None:
                    return idx, pos, self.__class__.shared_targets.get(idx)

        return None

    def _safe_fallback_toward_home(self, game_state):
        """Optimized safe fallback with single-pass evaluation."""
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP

        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP

        home_tiles = self._get_home_tiles(game_state)
        home_target = min(home_tiles, key=lambda t: self.get_maze_distance(my_pos, t)) if home_tiles else my_pos
        cur_dist_home = self.get_maze_distance(my_pos, home_target)

        # Get dangerous positions once
        dangerous_ghosts = self._get_visible_ghost_positions(game_state)

        def is_suicide(pos):
            return any(self.get_maze_distance(pos, gpos) <= 1 for gpos in dangerous_ghosts) if dangerous_ghosts and pos else False

        # Find best moves toward home
        candidates = []
        for action in legal:
            if action == Directions.STOP:
                continue

            succ = game_state.generate_successor(self.index, action)
            succ_pos = succ.get_agent_state(self.index).get_position()
            
            if succ_pos and not is_suicide(succ_pos):
                d_home = self.get_maze_distance(succ_pos, home_target)
                if d_home < cur_dist_home:
                    candidates.append(action)

        if candidates:
            return random.choice(candidates)

        # Try STOP if safe
        if Directions.STOP in legal and not is_suicide(my_pos):
            return Directions.STOP

        return Directions.STOP

    def _begin_trap_escape(self, game_state):
        """Optimized trap escape initiation."""
        self.in_trap_escape = True

        my_pos = game_state.get_agent_state(self.index).get_position()
        pocket = self.topology.pocket_id.get(my_pos)

        exits = list(self.topology.pocket_exits[pocket])
        door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))

        # Find safe neighbor efficiently
        neighbors = [(door[0] + dx, door[1] + dy) 
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]]

        walls = game_state.get_walls()
        safe_neighbors = [n for n in neighbors
                         if (0 <= n[0] < self.map_width 
                             and 0 <= n[1] < self.map_height
                             and not walls[n[0]][n[1]]
                             and self.topology.trap_depth(n) == 0)]

        escape_target = min(safe_neighbors, key=lambda n: self.get_maze_distance(my_pos, n)) if safe_neighbors else door

        self.trap_escape_path = self.pathfinder.find_path(
            game_state, my_pos, escape_target, avoid_enemies=False
        )

        return self.trap_escape_path.pop(0) if self.trap_escape_path else self._safe_fallback_toward_home(game_state)

    def _continue_trap_escape(self, game_state):
        """Follow escape path until fully out of trap."""
        my_pos = game_state.get_agent_state(self.index).get_position()

        # If we reached safety, stop escaping
        if self.topology.trap_depth(my_pos) == 0:
            self.in_trap_escape = False
            self.trap_escape_path = None
            return None  # Let normal BT take over

        # If path is empty, recompute
        if not self.trap_escape_path:
            pocket = self.topology.pocket_id.get(my_pos)
            try:
                exits = list(self.topology.pocket_exits[pocket])
            except Exception:
                # If pocket lookup fails, abort escape mode
                self.in_trap_escape = False
                self.trap_escape_path = None
                return self._safe_fallback_toward_home(game_state)
            
            door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))

            self.trap_escape_path = self.pathfinder.find_path(
                game_state, my_pos, door, avoid_enemies=False
            )

            if not self.trap_escape_path:
                return self._safe_fallback_toward_home(game_state)

        # Follow next step on persistent path
        return self.trap_escape_path.pop(0)
    

    #TODO: WHEN ON HOME SIDE, ADD OPTION TO GO FOR SUPER PELLET?
    #TODO: CHANGE RULE OF ALWAYS GOING HOME WHEN DIST == 1?
    #TODO: PALLET IN TRAP REGION LOGIC?
    #TODO: MAYBE WE CAN ADD FOOD LEVEL DANGER?