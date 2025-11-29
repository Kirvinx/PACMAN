import random
from collections import deque
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import manhattan_distance

# Behavior tree nodes
from contest.agents.team_name_1.baseline.behavior_tree import (
    Selector, Sequence, Condition, Action
)

# Utilities
from contest.agents.team_name_1.beliefline.exits import TerritoryAnalyzer
from contest.agents.team_name_1.beliefline.topology import MapTopologyAnalyzer
from contest.agents.team_name_1.beliefline.pathfinding_belief_opt import AStarPathfinder
from contest.agents.team_name_1.beliefline.pathfinding_defensive import DefensiveAStarPathfinder
from contest.agents.team_name_1.beliefline.belief_shared import GhostBeliefTracker
from contest.distance_calculator import Distancer

class TeamContext:
    def __init__(self, game_state, team_indices, opponents, is_red, agent):
        self.team_indices = tuple(sorted(team_indices))
        self.opponents = opponents
        self.is_red = is_red

        self.modes = {}           
        self.targets = {}
        self.interceptor = None
        self.group_assignment = {}
        self.last_assignment_turn = -1

        self.food_danger = {}   # (x, y) -> danger score
        self.last_on_enemy_side = {}      # idx -> bool
        self.double_defense_active = False

        # Distancer shared for this team+layout
        self.distancer = Distancer(game_state.data.layout)
        self.distancer.get_maze_distances()

        # Belief tracker shared for this team
        self.belief_tracker = GhostBeliefTracker(
            agent=agent,              # first agent becomes "owner" for debugging etc.
            opponents=opponents,
            team_indices=team_indices,
        )
        self.belief_tracker.initialize_uniformly(game_state)



class UnifiedBeliefBTAgent(CaptureAgent):
    """
    Unified agent that dynamically switches between offensive and defensive roles
    based on game state, using belief tracking and behavior trees.
    """
    _contexts = {}

    # Offensive constants
    CARRY_THRESHOLD = 10
    ESCAPE_PELLET_RISK_FACTOR = 0.9
    MIN_DEADLOCK_DIST = 2
    DEADLOCK_COMMIT = 10
    TRAP_ABORT_THRESHOLD = 5
    TIME_SAFETY_FACTOR = 1.5
    TICKS_PER_STEP = 4
    MAX_CYCLE_LEN = 6            # detect cycle lengths up to 6
    DEADLOCK_HISTORY_LEN = MAX_CYCLE_LEN * 3    # must be >= 3 * MAX_CYCLE_LEN

    DIR2DELTA = {
        Directions.NORTH: (0, 1),
        Directions.SOUTH: (0, -1),
        Directions.EAST:  (1, 0),
        Directions.WEST:  (-1, 0),
    }


    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        cls = self.__class__

        # ----- compute a unique key for this game+team -----
        game_id = id(game_state.data)  # unique per game instance
        team_indices = tuple(sorted(self.get_team(game_state)))
        ctx_key = (game_id, team_indices)

        # ----- create context if first time for this game+team -----
        if ctx_key not in cls._contexts:
            ctx = TeamContext(
                game_state=game_state,
                team_indices=team_indices,
                opponents=self.get_opponents(game_state),
                is_red=self.red,
                agent=self,   # first agent becomes the 'agent' for the tracker
            )
            cls._contexts[ctx_key] = ctx
        else:
            ctx = cls._contexts[ctx_key]
            # refresh belief tracker links for this game
            ctx.belief_tracker.debug_agent = self
            ctx.belief_tracker.opponents = self.get_opponents(game_state)
            ctx.belief_tracker.team_indices = self.get_team(game_state)
            ctx.belief_tracker.initialize_uniformly(game_state)

        # store on the instance
        self.ctx = ctx

        # Common per-agent setup (map, topology, pathfinders, etc.)
        self._init_common_components(game_state)

        # local per-agent state (deadlock, trap, etc.)
        self._init_offensive_state()
        self._init_defensive_state(game_state)
            
        # Build behavior trees
        self.offense_tree = self._build_offense_tree()
        self.defense_tree = self._build_defense_tree()
        
        # Build main behavior tree with mode selection
        self.behavior_tree = Selector([
            Sequence([
                Condition(lambda a, gs: a.mode == "offense"),
                Action(lambda a, gs: a.offense_tree.execute(a, gs))
            ]),
            Sequence([
                Condition(lambda a, gs: a.mode == "defense"),
                Action(lambda a, gs: a.defense_tree.execute(a, gs))
            ])
        ])
        
        # Start with offensive mode by default
        self.mode = "offense"


    def _init_common_components(self, game_state):
        """Initialize components shared between offensive and defensive modes."""
        # Cache game constants
        walls = game_state.get_walls()
        self.map_width = walls.width
        self.map_height = walls.height
        self.mid_x = self.map_width // 2
        
        # Team information from context
        self.team_indices = self.ctx.team_indices
        self.opponents = self.ctx.opponents
        
        # Initialize analyzers and pathfinders (per agent)
        self.topology = MapTopologyAnalyzer(walls)
        self.pathfinder = AStarPathfinder(self)
        self.defensive_pathfinder = DefensiveAStarPathfinder(self)
        
        # Initialize territory analyzer
        self.territory = TerritoryAnalyzer(game_state, self.red)
        self.home_entries = self._compute_home_entries(game_state)
        
        # Shared Distancer and BeliefTracker from context
        self.distancer = self.ctx.distancer
        self.belief_tracker = self.ctx.belief_tracker

        self._init_food_danger_cache(game_state)


    def _init_offensive_state(self):
        """Initialize offensive-specific state variables."""
        self.recent_positions = deque(maxlen=self.DEADLOCK_HISTORY_LEN)
        self.last_intent = None
        self.deadlock_target = None
        self.deadlock_commit = 0
        self.deadlock = 0
        self.in_trap_escape = False
        self.trap_escape_path = None

    def _init_defensive_state(self, game_state):
        """Initialize defensive-specific state variables."""
        self.patrol_target = None
        self._avoid_current_threat = None

    def choose_action(self, game_state):
        """Main action selection with dynamic role switching."""
        # Update mode based on game state
        self._update_mode(game_state)
      

        # Advance motion model (only lowest index agent)
        is_leader = self.index == min(self.team_indices)
        if is_leader:
            self.belief_tracker.elapse_time(game_state)
            if self._two_defenders_active(game_state):
                self._assign_groups_globally(game_state)
        
        # Observe with sonar
        self.belief_tracker.observe(game_state, self.index, is_leader)
        
        # Handle trap escape (offensive only)
        if self.mode == "offense" and self.in_trap_escape:
            action = self._continue_trap_escape(game_state)
            if action and action in game_state.get_legal_actions(self.index):
                return action
            self.in_trap_escape = False
            self.trap_escape_path = None
            return self._safe_fallback(game_state)
        
        # Update tracking for offensive mode
        if self.mode == "offense":
            my_pos = game_state.get_agent_state(self.index).get_position()
            if my_pos:
                self.recent_positions.append(my_pos)
        
        # Execute behavior tree
        action = self.behavior_tree.execute(self, game_state)
        legal = game_state.get_legal_actions(self.index)
        
        if action in legal:
            return action
        elif self.mode == "offense":
            return self._safe_fallback(game_state)
        else:
            return Directions.STOP

    def _update_mode(self, game_state):
        """
        Decide offense/defense roles for the whole team.

        Recomputed on every call. Modes are stored in ctx.modes and
        each agent just reads its own entry.
        """
        team = sorted(self.team_indices)

        # Ensure context fields exist (defensive check)
        if not hasattr(self.ctx, "last_on_enemy_side"):
            self.ctx.last_on_enemy_side = {}
        if not hasattr(self.ctx, "double_defense_active"):
            self.ctx.double_defense_active = False

        # 1) Score: are we winning?
        raw_score = game_state.get_score()
        we_winning = (raw_score > 0) if self.red else (raw_score < 0)

        # 2) Is there an enemy Pacman on our side?
        intruder_here = self._intruder_inside_or_past_entrance(game_state)

        # 3) Did any of *our* agents just come back from enemy side?
        any_just_returned = False
        for idx in team:
            st = game_state.get_agent_state(idx)
            pos = st.get_position() if st else None

            if pos is None:
                continue

            on_enemy = self._is_on_enemy_side_pos(game_state, pos)
            was_on_enemy = self.ctx.last_on_enemy_side.get(idx, on_enemy)

            # “just returned home from enemy side”
            if was_on_enemy and not on_enemy:
                any_just_returned = True

            self.ctx.last_on_enemy_side[idx] = on_enemy

        # 4) Update persistent "double defense" flag
        if self.ctx.double_defense_active:
            # Leave double-defense mode once the intruder is gone.
            if not intruder_here:
                print("ended")
                self.ctx.double_defense_active = False
        else:
            # Trigger double-defense when:
            #   - we are winning
            #   - there is an intruder
            #   - someone just came back from enemy side
            if we_winning and intruder_here and any_just_returned:
                print("True")
                self.ctx.double_defense_active = True

        # 5) Assign modes for this turn
        modes = {}

        if self.ctx.double_defense_active:
            # Both defend
            for idx in team:
                modes[idx] = "defense"
        else:
            # Default: 1 defender (lower index), 1 attacker (higher index)
            defender = team[0]
            attacker = team[1]
            modes[defender] = "defense"
            modes[attacker] = "offense"

        self.ctx.modes = modes

        # Everyone reads their mode
        self.mode = self.ctx.modes.get(self.index, "offense")


    # ==================== OFFENSIVE BEHAVIOR TREE ====================

    def _build_offense_tree(self):
        """Build offensive behavior tree."""
        return Selector([
            # Deadlock handling
            Sequence([
                Condition(lambda a, gs: a.deadlock_target is not None),
                Action(lambda a, gs: a._pursue_deadlock_target(gs))
            ]),
            Sequence([
                Condition(lambda a, gs: a._is_stuck()),
                Action(lambda a, gs: a.break_deadlock(gs))
            ]),
            # Time pressure
            Sequence([
                Condition(lambda a, gs: a._should_hurry_home(gs)),
                Action(lambda a, gs: a.return_home(gs))
            ]),
            # Retreat from danger
            Sequence([
                Condition(lambda a, gs: a._should_retreat(gs)),
                Action(lambda a, gs: a.run_away(gs))
            ]),
            # Carry limit reached
            Sequence([
                Condition(lambda a, gs: 
                    gs.get_agent_state(a.index).num_carrying >= a.CARRY_THRESHOLD),
                Action(lambda a, gs: a.return_home(gs))
            ]),
            # Food hunting
            Selector([
                Sequence([
                    Condition(lambda a, gs: a._two_offensive_agents(gs)),
                    Action(lambda a, gs: a._hunt_food(gs, dual=True))
                ]),
                Action(lambda a, gs: a._hunt_food(gs, dual=False))
            ])
        ])

    # ==================== DEFENSIVE BEHAVIOR TREE ====================

    def _build_defense_tree(self):
        """Build defensive behavior tree."""
        return Selector([
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

    # ==================== SHARED STATE CHECKS ====================

    def is_on_enemy_side(self, game_state):
        """Check if on opponent's half."""
        pos = game_state.get_agent_state(self.index).get_position()
        return pos and (pos[0] >= self.mid_x if self.red else pos[0] < self.mid_x)

    def _is_on_home_side_pos(self, game_state, pos):
        """Check if position is on home side."""
        if not pos:
            return True
        mid_x = game_state.get_walls().width // 2
        return pos[0] < mid_x if self.red else pos[0] >= mid_x

    def _home_border_x(self, game_state):
        """Get x-coordinate of home border."""
        mid_x = game_state.get_walls().width // 2
        return (mid_x - 1) if self.red else mid_x

    def _get_scared_timer(self, game_state):
        """Get current scared timer."""
        state = game_state.get_agent_state(self.index)
        if not state:
            return 0
        return getattr(state, "scared_timer", getattr(state, "scaredTimer", 0))

    def _is_scared(self, game_state):
        """Check if currently scared."""
        return self._get_scared_timer(game_state) > 0

    def is_power_active(self, game_state):
        """Check if any enemy is scared."""
        return any(game_state.get_agent_state(o).scared_timer > 3 for o in self.opponents)

    def _is_dangerous_ghost(self, opp_state):
        """Check if enemy ghost is dangerous."""
        return not opp_state.is_pacman and opp_state.scared_timer <= 2

    # ==================== OFFENSIVE METHODS ====================

    def _is_stuck(self):
        """
        Detect movement deadlocks by checking if the last positions
        consist of a short cycle repeated at least 3 times.

        Examples that return True:
        - AAA                  (k=1, A|A|A)
        - ABABABAB             (k=2, AB|AB|AB in the tail)
        - ABCABCABC            (k=3, ABC|ABC|ABC)
        - ABCDABCDABCD         (k=4, ABCD|ABCD|ABCD)
        - (extended) 6-length cycles if history >= 18
        """
        pos = list(self.recent_positions)
        n = len(pos)

        # allow cycle lengths up to MAX_CYCLE_LEN, but not more than n//3
        max_cycle_len = min(self.MAX_CYCLE_LEN, n // 3)
        if max_cycle_len == 0:
            return False

        for k in range(1, max_cycle_len + 1):
            needed = 3 * k
            if n < needed:
                continue

            start = n - needed
            block1 = pos[start : start + k]
            block2 = pos[start + k : start + 2 * k]
            block3 = pos[start + 2 * k : start + 3 * k]

            if block1 == block2 == block3:
                return True

        return False


    def _two_offensive_agents(self, game_state):
        """Check if 2+ agents are offensive."""
        return sum(1 for i in self.team_indices 
                  if self.ctx.modes.get(i, "offense") == "offense") >= 2

    def _should_retreat(self, game_state):
        pos = game_state.get_agent_state(self.index).get_position()
        if not pos:
            return False

        # first, normal logic outside trap
        if not self.topology.is_in_trap_region(pos):
            return (self.is_on_enemy_side(game_state) and
                    not self.is_power_active(game_state) and
                    self.ghost_nearby(game_state, 5))

        # ---------- TRAP logic ----------
        # inside a trap: even scared ghosts matter
        pocket = self.topology.pocket_id.get(pos)
        exits = list(self.topology.pocket_exits[pocket])
        door = min(exits, key=lambda e: self.get_maze_distance(pos, e))

        # Pacman's time to escape
        pac_time = self.get_maze_distance(pos, door)

        # Ghost time to become dangerous at door
        min_ghost_time = float("inf")
        for opp in self.opponents:
            st = game_state.get_agent_state(opp)
            gpos = st.get_position() if st else None
            if gpos:
                t = self._ghost_danger_time(game_state, gpos, st, door)
                min_ghost_time = min(min_ghost_time, t)

        # Retreat if we can't escape in time
        return pac_time >= min_ghost_time

    def ghost_nearby(self, game_state, radius):
        """Check if dangerous ghost within radius."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return False
        
        for opp in self.opponents:
            st = game_state.get_agent_state(opp)
            if self._is_dangerous_ghost(st):
                pos = st.get_position()
                if pos and self.get_maze_distance(my_pos, pos) <= radius:
                    return True
        return False


    def break_deadlock(self, game_state):
        """Handle deadlock situations."""
        history = list(self.recent_positions)
        self.recent_positions.clear()
        
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP
        
        if self.is_on_enemy_side(game_state):
            cycle_crosses_home = any(
                self._is_on_home_side_pos(game_state, p) for p in history
            )

            # If cycle crossed home-side, treat this like defensive deadlock
            if cycle_crosses_home:
                border_x = self.mid_x - 1 if self.red else self.mid_x
                border_tiles = [(border_x, y) for y in range(self.map_height)
                                if not game_state.get_walls()[border_x][y]]

                far_tiles = [t for t in border_tiles
                            if self.get_maze_distance(my_pos, t) >= self.MIN_DEADLOCK_DIST]

                target = random.choice(far_tiles or border_tiles)
                self.deadlock_target = target
                self.deadlock_commit = self.DEADLOCK_COMMIT

                path = self.pathfinder.find_path(
                    game_state, my_pos, target,
                    avoid_enemies=True     
                )
                if path:
                    return path[0]
            
            if self.last_intent != "run_home" and self.deadlock == 0:
                self.deadlock = 1
                return self._escape_best_option(game_state)
            
            target = self._pick_deep_target(game_state, my_pos)
            if target:
                self.deadlock_target = target
                self.deadlock_commit = 6
                path = self.pathfinder.find_path(game_state, my_pos, target, avoid_enemies=True)
                if path:
                    return path[0]
        else:
            border_x = self.mid_x - 1 if self.red else self.mid_x
            border_tiles = [(border_x, y) for y in range(self.map_height) 
                          if not game_state.get_walls()[border_x][y]]
            
            far_tiles = [t for t in border_tiles 
                        if self.get_maze_distance(my_pos, t) >= self.MIN_DEADLOCK_DIST]
            target = random.choice(far_tiles or border_tiles)
            
            self.deadlock_target = target
            self.deadlock_commit = self.DEADLOCK_COMMIT
            
            path = self.defensive_pathfinder.find_path(
                game_state, my_pos, target,
                avoid_path_tiles=None,       
                lambda_path_overlap=100.0
            )
            if path:
                return path[0]
        
        self._clear_deadlock()
        return self._safe_fallback(game_state)

    def _pursue_deadlock_target(self, game_state):
        """Continue pursuing deadlock target."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos or not self.deadlock_target or self.deadlock_commit <= 0:
            self._clear_deadlock()
            return Directions.STOP
        
        if self.is_on_enemy_side(game_state):
            # Offensive-side deadlock pursuit uses offensive A*
            path = self.pathfinder.find_path(
                game_state, my_pos, self.deadlock_target,
                avoid_enemies=True       # offensive pathfinder signature stays the same
            )
        else:
            # Defensive-side deadlock pursuit uses NEW defensive A*
            path = self.defensive_pathfinder.find_path(
                game_state, my_pos, self.deadlock_target,
                avoid_path_tiles=None,   # No teammate avoidance in deadlock mode
                lambda_path_overlap=0.0  # No penalty
            )
                
        if path:
            self.deadlock_commit -= 1
            if self.get_maze_distance(my_pos, self.deadlock_target) <= 1:
                self._clear_deadlock()
            return path[0]
        
        self._clear_deadlock()
        return self._safe_fallback(game_state)

    def _clear_deadlock(self):
        """Clear deadlock state."""
        self.deadlock_target = None
        self.deadlock_commit = 0
        self.deadlock = 0

    def _pick_deep_target(self, game_state, my_pos):
        """Pick target deep in enemy territory."""
        enemy_min_x = self.mid_x if self.red else 0
        enemy_max_x = self.map_width - 1 if self.red else self.mid_x - 1
        
        food_list = self.get_food(game_state).as_list()
        depth_threshold = my_pos[0] + 3 if self.red else my_pos[0] - 3
        deeper_food = [f for f in food_list 
                      if enemy_min_x <= f[0] <= enemy_max_x and
                      (f[0] >= depth_threshold if self.red else f[0] <= depth_threshold)]
        
        if deeper_food:
            return random.choice(deeper_food)
        
        walls = game_state.get_walls()
        enemy_tiles = [(x, y) for x in range(enemy_min_x, enemy_max_x + 1)
                      for y in range(self.map_height) if not walls[x][y]]
        return random.choice(enemy_tiles) if enemy_tiles else None

    def _hunt_food(self, game_state, dual):
        """Unified food hunting."""
        self.last_intent = "food"
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP
        
        if self._should_abort_trap(game_state, my_pos):
            return self._begin_trap_escape(game_state)
        
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return self.return_home(game_state)
        
        target = self._pick_food_target(game_state, my_pos, food_list, dual)
        if not target:
            return self.return_home(game_state) if dual and self.is_on_enemy_side(game_state) else Directions.STOP
        
        self.ctx.targets[self.index] = target
        
        if (self.topology.is_in_trap_region(target) and 
            not self._safe_to_enter_trap(game_state, my_pos, target)):
            target = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        
        path = self.pathfinder.find_path(game_state, my_pos, target, avoid_enemies=True)
        return path[0] if path else self._safe_fallback(game_state)

    def _pick_food_target(self, game_state, my_pos, food_list, dual):
        """Pick food target based on mode."""
        if not food_list:
            return None
        
        if dual:
            mate_info = self._get_teammate_info(game_state)
            if not mate_info:
                return self._pick_food_single(game_state, my_pos, food_list)
            
            _, mate_pos, mate_target = mate_info
            ref_pos = mate_target or mate_pos
            
            non_overlap = [f for f in food_list 
                         if self.get_maze_distance(f, ref_pos) >= 3]
            if not non_overlap:
                return None
            
            return min(non_overlap, key=lambda f: 
                      self.get_maze_distance(my_pos, f) - 0.6 * self.get_maze_distance(ref_pos, f))
        else:
            return self._pick_food_single(game_state, my_pos, food_list)

    def _pick_food_single(self, game_state, my_pos, food_list):
        """Single agent food selection."""
        if self.is_on_enemy_side(game_state):
            return min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        
        ghosts = self.get_believed_ghost_positions(game_state)
        
        def score(f):
            cluster = sum(1 for f2 in food_list if manhattan_distance(f, f2) <= 3)
            risk = self.topology.trap_depth(f)
            g_dist = min((self.get_maze_distance(f, g) for g in ghosts), default=10)
            return cluster + 3.0 * g_dist - 2.0 * risk
        
        candidates = sorted([(score(f), f) for f in food_list], reverse=True, key=lambda x: x[0])
        
        scores = [s for s, _ in candidates]
        weights = [2.71828 ** (0.5 * (s - min(scores))) for s in scores]
        
        r = random.uniform(0, sum(weights))
        cumulative = 0
        for (s, f), w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return f
        return candidates[0][1]

    def _safe_to_enter_trap(self, game_state, my_pos, food_pos):
        """Check if safe to enter trap for food."""
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
        
        ghosts = self._get_visible_ghosts(game_state)
        return all(self.get_maze_distance(g, door) > required_dist for g in ghosts)

    def _should_abort_trap(self, game_state, my_pos):
        pocket = self.topology.pocket_id.get(my_pos)
        if not pocket or pocket not in self.topology.off_pockets:
            return False

        exits = list(self.topology.pocket_exits[pocket])
        door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))

        pac_time = self.get_maze_distance(my_pos, door)

        # default: allow a configurable extra margin if you want
        pac_time += self.TRAP_ABORT_THRESHOLD

        min_ghost_time = float("inf")

        for opp in self.opponents:
            st = game_state.get_agent_state(opp)
            gpos = st.get_position() if st else None
            if gpos:
                t = self._ghost_danger_time(game_state, gpos, st, door)
                min_ghost_time = min(min_ghost_time, t)

        if min_ghost_time == float("inf"):
            return False

        return pac_time >= min_ghost_time

    
    def _ghost_danger_time(self, game_state, ghost_pos, ghost_state, door):
        if not ghost_state or ghost_state.is_pacman or not ghost_pos:
            return float("inf")

        g_dist = self.get_maze_distance(ghost_pos, door)

        # seconds/ticks → convert to steps
        scared = ghost_state.scared_timer
        scared_steps = scared  if scared > 0 else 0
        # Ghost becomes lethal at door after:
        #   1. reaching door (g_dist)
        #   2. no longer scared (scared_steps)
        return max(g_dist, scared_steps)

    def _begin_trap_escape(self, game_state):
        """Start trap escape sequence."""
        self.in_trap_escape = True
        my_pos = game_state.get_agent_state(self.index).get_position()
        pocket = self.topology.pocket_id.get(my_pos)
        
        exits = list(self.topology.pocket_exits[pocket])
        door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))
        
        neighbors = [(door[0] + dx, door[1] + dy) for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]]
        walls = game_state.get_walls()
        safe = [n for n in neighbors 
               if 0 <= n[0] < self.map_width and 0 <= n[1] < self.map_height
               and not walls[n[0]][n[1]] and self.topology.trap_depth(n) == 0]
        
        target = min(safe, key=lambda n: self.get_maze_distance(my_pos, n)) if safe else door
        
        self.trap_escape_path = self.pathfinder.find_path(
            game_state, my_pos, target, avoid_enemies=False
        )
        
        if self.trap_escape_path:
            return self.trap_escape_path.pop(0)
        
        return self._greedy_step(game_state, door) or self._safe_fallback(game_state)

    def _continue_trap_escape(self, game_state):
        """Continue trap escape."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        if self.topology.trap_depth(my_pos) == 0:
            self.in_trap_escape = False
            self.trap_escape_path = None
            return None
        
        if not self.trap_escape_path:
            pocket = self.topology.pocket_id.get(my_pos)
            try:
                exits = list(self.topology.pocket_exits[pocket])
                door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))
                self.trap_escape_path = self.pathfinder.find_path(
                    game_state, my_pos, door, avoid_enemies=False
                )
            except:
                self.in_trap_escape = False
                return self._safe_fallback(game_state)
            
            if not self.trap_escape_path:
                return self._greedy_step(game_state, door) or self._safe_fallback(game_state)
        
        return self.trap_escape_path.pop(0)

    def _greedy_step(self, game_state, target):
        """Take greedy step toward target."""
        legal = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos or not target or not legal:
            return None
        
        best_dist = self.get_maze_distance(my_pos, target)
        best_action = None
        
        for action in legal:
            if action == Directions.STOP:
                continue
            succ = game_state.generate_successor(self.index, action)
            succ_pos = succ.get_agent_state(self.index).get_position()
            if succ_pos:
                d = self.get_maze_distance(succ_pos, target)
                if d < best_dist:
                    best_dist = d
                    best_action = action
        
        return best_action

    def return_home(self, game_state):
        """Return to home side."""
        self.last_intent = "run_home"
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP
        
        home_tiles = self._get_home_tiles(game_state)
        if not home_tiles:
            return Directions.STOP
        
        nearest = min(home_tiles, key=lambda t: self.get_maze_distance(my_pos, t))
        path = self.pathfinder.find_path(game_state, my_pos, nearest, avoid_enemies=True)
        return path[0] if path else self._safe_fallback(game_state)

    def run_away(self, game_state):
        """Escape from danger."""
        self.last_intent = "escape"
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP
        
        # Trap abort logic stays as is
        if self._should_abort_trap(game_state, my_pos):
            return self._begin_trap_escape(game_state)
        
        close_ghosts = []
        min_ghost_dist = None

        # Collect visible dangerous ghosts within radius 5
        for opp in self.opponents:
            st = game_state.get_agent_state(opp)
            if self._is_dangerous_ghost(st):
                pos = st.get_position()
                if pos:
                    dist = self.get_maze_distance(my_pos, pos)
                    if dist <= 5:
                        close_ghosts.append(pos)
                        if min_ghost_dist is None or dist < min_ghost_dist:
                            min_ghost_dist = dist

        # === NEW PART 1: "no safe progress" → always go home ===
        # If we are being chased (ghost in [1..5]) and there is no
        # consumable that is safe under the formula, we should just bail.
        if close_ghosts and min_ghost_dist is not None:
            if not self._has_safe_progress_consumable_for_chase(game_state, min_ghost_dist):
                return self._escape_best_option(game_state)

        carrying = game_state.get_agent_state(self.index).num_carrying
        p_home = min(1.0, float(carrying) / max(1, self.CARRY_THRESHOLD))
        
        # Existing aggressive home decision still applies
        """
        if random.random() < p_home:
            print("huh")
            return self._escape_best_option(game_state)
        """
        
         # === NEW PART 2: when kiting food/capsules under chase ===
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)

        # If we see a chaser in visible radius, use danger-aware logic
        if close_ghosts and min_ghost_dist is not None:
            # 1) Filter for safe food given current chase distance
            safe_food = []
            for f in food_list:
                danger = self.get_food_danger(f)
                if self._is_danger_safe_with_chaser(danger, min_ghost_dist):
                    safe_food.append(f)

            # 1a) If there is any safe food, kite towards that (ignore capsules)
            if safe_food:
                weights = [
                    (f, max(1, self.get_maze_distance(my_pos, f)) ** 2)
                    for f in safe_food
                ]
                total = sum(w for _, w in weights)
                if total > 0:
                    r = random.uniform(0, total)
                    acc = 0
                    for f, w in weights:
                        acc += w
                        if r <= acc:
                            path = self.pathfinder.find_path(
                                game_state, my_pos, f, avoid_enemies=True
                            )
                            if path:
                                return path[0]
                            break

                # If for some reason we failed to path to safe food, bail home
                return self._escape_best_option(game_state)

            # 1b) No safe food. If there is at least one capsule, that is the
            #     only way to make progress → we MUST go for a capsule.
            if capsules:
                # simple choice: nearest capsule
                target_capsule = min(
                    capsules, key=lambda c: self.get_maze_distance(my_pos, c)
                )
                path = self.pathfinder.find_path(
                    game_state, my_pos, target_capsule, avoid_enemies=True
                )
                if path:
                    return path[0]

                # Capsule exists but no path? Then just bail.
                return self._escape_best_option(game_state)

            # 1c) No safe food and no capsules → no progress possible → go home
            return self._escape_best_option(game_state)

        # === Not actively chased (no close_ghosts) → old food-kiting behaviour ===
        if food_list:
            weights = [
                (f, max(1, self.get_maze_distance(my_pos, f)) ** 2)
                for f in food_list
            ]
            total = sum(w for _, w in weights)
            if total > 0:
                r = random.uniform(0, total)
                acc = 0
                for f, w in weights:
                    acc += w
                    if r <= acc:
                        path = self.pathfinder.find_path(
                            game_state, my_pos, f, avoid_enemies=True
                        )
                        if path:
                            return path[0]
                        break

        # No good food target → escape towards home/pellet
        return self._escape_best_option(game_state)


    def _escape_best_option(self, game_state):
        """Find best escape route."""
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
        
        return self._safe_fallback(game_state)

    def _safe_fallback(self, game_state):
        """Safe fallback movement."""
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP
        
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return Directions.STOP
        
        home_tiles = self._get_home_tiles(game_state)
        home_target = min(home_tiles, key=lambda t: self.get_maze_distance(my_pos, t)) if home_tiles else my_pos
        cur_dist = self.get_maze_distance(my_pos, home_target)
        
        ghosts = self._get_visible_ghosts(game_state)
        is_suicide = lambda p: any(self.get_maze_distance(p, g) <= 1 for g in ghosts) if ghosts else False
        
        candidates = []
        for action in legal:
            if action == Directions.STOP:
                continue
            succ = game_state.generate_successor(self.index, action)
            succ_pos = succ.get_agent_state(self.index).get_position()
            if succ_pos and not is_suicide(succ_pos):
                if self.get_maze_distance(succ_pos, home_target) < cur_dist:
                    candidates.append(action)
        
        if candidates:
            return random.choice(candidates)
        
        action = self._panic_move(game_state, ghosts)
        if action:
            return action
        
        return Directions.STOP if Directions.STOP in legal and not is_suicide(my_pos) else Directions.STOP

    def _panic_move(self, game_state, ghosts):
        """Panic movement away from ghosts."""
        if not ghosts:
            return None
        
        legal = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos or not legal:
            return None
        
        home_progress = lambda p: 1 if (p[0] < my_pos[0] if self.red else p[0] > my_pos[0]) else 0
        
        best_score = float("-inf")
        best_action = None
        
        for action in legal:
            if action == Directions.STOP:
                continue
            
            succ = game_state.generate_successor(self.index, action)
            succ_pos = succ.get_agent_state(self.index).get_position()
            if not succ_pos:
                continue
            
            ghost_dist = min(self.get_maze_distance(succ_pos, g) for g in ghosts)
            trap_penalty = 5 * self.topology.trap_depth(succ_pos)
            score = 3.0 * ghost_dist + home_progress(succ_pos) - trap_penalty
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

    def _should_hurry_home(self, game_state):
        """Check if need to hurry home due to time pressure."""
        if not self.is_on_enemy_side(game_state):
            return False

        food_list = self.get_food(game_state).as_list()
        if len(food_list) <= 2 and self.is_on_enemy_side(game_state):
            return True

        time_left = getattr(game_state.data, "timeleft", None)
        if time_left is None:
            return False

        my_pos = game_state.get_agent_state(self.index).get_position()
        if not my_pos:
            return False

        home_tiles = self._get_home_tiles(game_state)
        if not home_tiles:
            return False

        best_path_len = None

        for t in home_tiles:
            result = self.pathfinder.find_path(
                game_state, my_pos, t,
                avoid_enemies=True,
                return_cost=True
            )
            if not result:
                continue
            path, cost = result
            if path:
                dist = len(path)
                if best_path_len is None or dist < best_path_len:
                    best_path_len = dist

        if best_path_len is None:
            return False

        required_time = best_path_len * self.TIME_SAFETY_FACTOR * self.TICKS_PER_STEP

        return required_time >= time_left

    def _is_on_enemy_side_pos(self, game_state, pos):
        if not pos:
            return False
        mid_x = game_state.get_walls().width // 2
        return pos[0] >= mid_x if self.red else pos[0] < mid_x

    def _init_food_danger_cache(self, game_state):
        """
        Precompute per-tile danger values based purely on topology
        and store in the shared TeamContext. This is layout-static.
        """
        # If another agent already did this for the team, reuse it
        if getattr(self.ctx, "food_danger", None):
            return

        danger = {}
        walls = game_state.get_walls()
        W, H = walls.width, walls.height

        for x in range(W):
            for y in range(H):
                if walls[x][y]:
                    continue
                pos = (x, y)

                # Only care about enemy side tiles for offensive food
                if not self._is_on_enemy_side_pos(game_state, pos):
                    continue

                depth = self.topology.trap_depth(pos)
                if depth > 0:
                    danger[pos] = depth  # you can add scaling later if you want

        self.ctx.food_danger = danger

    def get_food_danger(self, pos):
        """Fast lookup: danger level of a given food position."""
        return self.ctx.food_danger.get(pos, 0)
    
    def _is_danger_safe_with_chaser(self, danger, chaser_dist):
        """
        Returns True if a consumable with given `danger` is safe to pursue
        while a chasing ghost is `chaser_dist` steps away.

        Rule:
            - if danger == 0 → always safe
            - if danger >= 1 → safe iff chaser_dist >= 2*danger + 1
        """
        if danger <= 0:
            return True
        return chaser_dist >= 2 * danger + 1

    def _has_safe_progress_consumable_for_chase(self, game_state, min_ghost_dist):
        """
        Returns True if, at current chase distance, there is any consumable
        (food or capsule) that is realistically safe/worth pursuing.

        - Food:
            danger d is safe iff _is_danger_safe_with_chaser(d, min_ghost_dist) is True.
        - Capsules:
            treated as always preserving progress potential, even if in traps.
        """
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)

        # Capsules "solve the problem" even if they're in traps.
        if capsules:
            return True

        if not food_list:
            return False

        for f in food_list:
            danger = self.get_food_danger(f)
            if self._is_danger_safe_with_chaser(danger, min_ghost_dist):
                return True

        return False


    # ==================== DEFENSIVE METHODS ====================

    def _entrance_x(self):
        """Get x-coordinate of defensive entrance line."""
        if self.territory.entrances:
            return self.territory.entrances[0][0]
        return self.mid_x - 1 if self.red else self.mid_x

    def _is_past_entrance_line(self, pos):
        """Check if position is past entrance line."""
        if not pos:
            return False

        ex = self._entrance_x()

        if self.red:
            return pos[0] < ex
        else:
            return pos[0] > ex

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

    def _closest_visible_intruder(self, game_state, max_manhattan=5):
        """Find closest visible intruder."""
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
        """Check if 2+ defenders active."""
        count = sum(1 for idx in self.team_indices 
                   if self.ctx.modes.get(idx) == "defense")
        return count >= 2

    def _am_primary_interceptor(self, game_state):
        """Check if we're the primary interceptor."""
        # TODO: implement proper interceptor selection for multiple defenders
        return True

    def _avoid_scared_invader(self, game_state, threat_info):
        """Avoid invader when scared."""
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        if not my_pos:
            return Directions.STOP

        _, threat_pos, _ = threat_info
        dist = self.get_maze_distance(my_pos, threat_pos)

        if dist == 2 and Directions.STOP in legal:
            return Directions.STOP

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

        if dist < 2:
            candidates.sort(
                key=lambda c: (
                    -c[2],
                    dist_to_closest_entrance(c[1]),
                    border_distance(c[1]),
                    str(c[0])
                )
            )
        elif dist > 2:
            candidates.sort(
                key=lambda c: (
                    c[2],
                    dist_to_closest_entrance(c[1]),
                    border_distance(c[1]),
                    str(c[0])
                )
            )

        my_depth = self.topology.trap_depth(my_pos)
        inv_depth = self.topology.trap_depth(threat_pos)

        safe = []
        for c in candidates:
            action, sp, new_dist = c
            next_depth = self.topology.trap_depth(sp)

            if my_depth > 0 and next_depth > my_depth and inv_depth <= my_depth:
                continue

            safe.append(c)

        return (safe or candidates)[0][0]

    def _intercept_invader(self, game_state):
        if self._is_scared(game_state):
            threat_info = self._closest_visible_intruder(game_state, max_manhattan=5)
            if threat_info:
                return self._avoid_scared_invader(game_state, threat_info)

        my_pos = game_state.get_agent_position(self.index)
        if not my_pos:
            return Directions.STOP

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

        self.ctx.targets[self.index] = target
        self.ctx.interceptor = self.index

        avoid_tiles = self._predicted_teammate_path_tiles(game_state, target)

        path = self.defensive_pathfinder.find_path(
            game_state, my_pos, target,
            avoid_path_tiles=avoid_tiles,
            lambda_path_overlap=100.0,
        )

        legal = game_state.get_legal_actions(self.index)
        if path and path[0] in legal:
            return path[0]

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

        if self.ctx.interceptor == self.index:
            self.ctx.interceptor = None

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

        path = self.defensive_pathfinder.find_path(
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
        my_group = self.ctx.group_assignment.get(self.index)
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

        path = self.defensive_pathfinder.find_path(
            game_state, my_pos, patrol_target
        )

        if path and path[0] in legal:
            return path[0]
        return Directions.STOP



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
            if idx != self.index and self.ctx.modes.get(idx) == "defense":
                return (
                    idx,
                    game_state.get_agent_position(idx),
                    self.ctx.targets.get(idx),
                )
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
        """Assign defenders to entrance groups."""
        group_a, group_b = getattr(self.territory, "entrance_groups", (None, None))
        if not group_a or not group_b:
            return

        defenders = [
            (idx, game_state.get_agent_position(idx))
            for idx in self.team_indices
            if self.ctx.modes.get(idx) == "defense"
            and game_state.get_agent_position(idx)
        ]

        if len(defenders) < 2:
            return

        defenders.sort(key=lambda t: t[0])
        (idx1, pos1), (idx2, pos2) = defenders[:2]

        def dist_to_group(pos, group):
            entrances = self.territory.entrances
            return min(self.get_maze_distance(pos, entrances[gi]) for gi in group)

        cost1 = dist_to_group(pos1, group_a) + dist_to_group(pos2, group_b)
        cost2 = dist_to_group(pos1, group_b) + dist_to_group(pos2, group_a)

        assignment = {idx1: "A", idx2: "B"} if cost1 <= cost2 else {idx1: "B", idx2: "A"}
        self.ctx.group_assignment = assignment

    # ==================== HELPER METHODS ====================

    def _get_home_tiles(self, game_state):
        """Get home side border tiles."""
        walls = game_state.get_walls()
        x = self.mid_x - 1 if self.red else self.mid_x
        return [(x, y) for y in range(self.map_height) if not walls[x][y]]

    def _get_visible_ghosts(self, game_state):
        """Get visible dangerous ghost positions."""
        positions = []
        for opp in self.opponents:
            st = game_state.get_agent_state(opp)
            if self._is_dangerous_ghost(st):
                pos = st.get_position()
                if pos:
                    positions.append(pos)
        return positions

    def get_believed_ghost_positions(self, game_state):
        """Get believed ghost positions from tracker."""
        positions = []
        for opp in self.opponents:
            belief = self.belief_tracker.get_belief(opp)
            if belief:
                positions.append(max(belief.items(), key=lambda kv: kv[1])[0])
            else:
                pos = game_state.get_agent_state(opp).get_position()
                if pos:
                    positions.append(pos)
        return positions

    def _get_teammate_info(self, game_state):
        """Get offensive teammate info."""
        for idx in self.team_indices:
            if idx != self.index and self.ctx.modes.get(idx, "offense") == "offense":
                st = game_state.get_agent_state(idx)
                if st:
                    pos = st.get_position()
                    if pos:
                        return idx, pos, self.ctx.targets.get(idx)
        return None

    def _path_to_positions(self, start_pos, actions):
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
        if not self._two_defenders_active(game_state):
            return None

        teammate_info = self._get_defensive_teammate_info(game_state)
        if not teammate_info:
            return None

        mate_idx, mate_pos, mate_target = teammate_info
        my_pos = game_state.get_agent_position(self.index)

        if not my_pos or not mate_pos or not target:
            return None

        d_self = self.get_maze_distance(my_pos, target)
        d_mate = self.get_maze_distance(mate_pos, target)

        i_am_leader = (
            d_self < d_mate or
            (d_self == d_mate and self.index < mate_idx)
        )

        if i_am_leader:
            return None

        mate_path = self.defensive_pathfinder.find_path(
            game_state,
            start=mate_pos,
            goal=target
        )

        if not mate_path:
            return None

        tiles = self._path_to_positions(mate_pos, mate_path)

        if tiles:
            tiles = tiles[:-1]

        return set(tiles)


#TODO: AFTERWARDS, BETTER SWITCHING BETWEEN THEM
#TODO: CONSIDER ON OFFENSE:
"""
4. Dynamic carry threshold instead of a fixed CARRY_THRESHOLD
6. Smarter “food escape” when running away (specifically ghost proximity)
"""


#TODO: PACMAN KILLING HIMSELF GOING INTO THEIR TERRITORY
