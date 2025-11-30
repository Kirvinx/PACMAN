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
    """

    CARRY_THRESHOLD = 10
    ESCAPE_PELLET_RISK_FACTOR = 1.5
    # shared across both instances of this class
    shared_targets = {}  # maps agent_index -> (x, y)
    shared_modes = {}     # agent_index -> "offense" / "defense"

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        self.team_indices = self.get_team(game_state)
        self.topology = MapTopologyAnalyzer(game_state.get_walls())
        self.belief_tracker = GhostBeliefTracker(self, self.get_opponents(game_state))
        self.belief_tracker.initialize_uniformly(game_state)
        self.pathfinder = AStarPathfinder(self)
        self.mode = "offense"

        # --- OFFENSIVE SUBTREE ---
        self.offense_tree = Selector([
            # 1) Emergency retreat if ghost nearby on enemy side
            Sequence([
                Condition(lambda agent, gs:
                    agent.is_on_enemy_side(gs)
                    and not agent.is_power_active(gs)
                    and agent.ghost_nearby(gs, radius=5)),
                Action(lambda agent, gs: agent.run_away(gs)),
            ]),
            # 2) Return home if carrying too much
            Sequence([
                Condition(lambda agent, gs:
                    gs.get_agent_state(agent.index).num_carrying >= agent.CARRY_THRESHOLD),
                Action(lambda agent, gs: agent.return_home(gs)),
            ]),
            # 3) Food-hunting: dual vs solo offense
            Selector([
                # dual-offense case
                Sequence([
                    Condition(lambda agent, gs: agent._two_offensive_agents(gs)),
                    Action(lambda agent, gs: agent.hunt_food_dual(gs)),
                ]),
                # fallback: solo offense
                Action(lambda agent, gs: agent.hunt_food_single(gs)),
            ]),
        ])

        # --- TOP-LEVEL BT (for now only offense; defense later) ---
        self.behavior_tree = Selector([
            Sequence([
                Condition(lambda agent, gs: agent.mode == "offense"),
                Action(lambda agent, gs: agent.offense_tree.execute(agent, gs)),
            ]),
            # later you’ll add a defense subtree here
        ])

    def choose_action(self, game_state):
        # Update beliefs
        self.belief_tracker.elapse_time(game_state)
        self.belief_tracker.observe(game_state)

        # Record our mode globally
        BeliefBTOffensiveAgent.shared_modes[self.index] = self.mode

        # Execute behavior tree
        action = self.behavior_tree.execute(self, game_state)
        
        legal = game_state.get_legal_actions(self.index)
        return action if action in legal else (random.choice(legal) if legal else Directions.STOP)

    def is_on_enemy_side(self, game_state):
        """Check if we're on opponent's half of the map."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return False
        mid_x = game_state.get_walls().width // 2
        return my_pos[0] >= mid_x if self.red else my_pos[0] < mid_x

    def is_power_active(self, game_state):
        """Check if any enemy ghost is scared."""
        return any(game_state.get_agent_state(opp).scared_timer > 0 
                  for opp in self.get_opponents(game_state))

    def _is_dangerous_ghost(self, opp_state):
        """Check if enemy ghost is dangerous (not scared, not pacman)."""
        return not opp_state.is_pacman and opp_state.scared_timer <= 2

    def _two_offensive_agents(self, game_state):
        """
        Returns True if at least two of our team agents are currently in offensive mode.
        (We use shared_modes, which RL / logic will fill each turn.)
        """
        offensive = [
            idx for idx in self.team_indices
            if BeliefBTOffensiveAgent.shared_modes.get(idx, "offense") == "offense"
        ]
        return len(offensive) >= 2


    def ghost_nearby(self, game_state, radius=5):
        """Check if dangerous ghost is within radius."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return False

        for opp in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp)
            opp_pos = opp_state.get_position()
            
            if (opp_pos is not None 
                and self._is_dangerous_ghost(opp_state)
                and self.get_maze_distance(my_pos, opp_pos) <= radius):
                return True
        return False

    def _get_visible_ghost_positions(self, game_state):
        """Get positions of visible dangerous ghosts."""
        positions = []
        for opp in self.get_opponents(game_state):
            st = game_state.get_agent_state(opp)
            pos = st.get_position()
            if pos is not None and self._is_dangerous_ghost(st):
                positions.append(pos)
        return positions

    def get_believed_ghost_positions(self, game_state):
        """Get most likely ghost positions from belief system."""
        positions = []
        for opp in self.get_opponents(game_state):
            belief_dist = self.belief_tracker.get_belief(opp)
            if belief_dist:
                best_pos = max(belief_dist.items(), key=lambda kv: kv[1])[0]
                positions.append(best_pos)
            else:
                opp_pos = game_state.get_agent_state(opp).get_position()
                if opp_pos is not None:
                    positions.append(opp_pos)
        return positions

    def _get_home_tiles(self, game_state):
        """Get list of tiles on our side of the border."""
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        x = mid_x - 1 if self.red else mid_x
        return [(x, y) for y in range(walls.height) if not walls[x][y]]

    def _pick_food_target_single(self, game_state, my_pos, food_list):
        """Choose food target based on current side."""
        if not food_list:
            return self.return_home(game_state)
            
        if self.is_on_enemy_side(game_state):
            # On enemy side: pick nearest
            return min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        
        # On home side: use scoring system
        believed_ghosts = self.get_believed_ghost_positions(game_state)
        scored = []
        
        for food_pos in food_list:
            # Cluster bonus
            cluster_size = sum(1 for f in food_list 
                             if manhattan_distance(food_pos, f) <= 3) - 1
            
            # Risk from topology
            risk_depth = self.topology.trap_depth(food_pos)
            
            # Ghost distance
            ghost_dist = (min(self.get_maze_distance(food_pos, gpos) 
                            for gpos in believed_ghosts) 
                         if believed_ghosts else 10)
            
            score = 3.0 * cluster_size + 1.0 * ghost_dist - 2.0 * risk_depth
            scored.append((score, food_pos))
        
        # Random choice from top candidates
        scored.sort(reverse=True, key=lambda x: x[0])
        k = min(5, len(scored))
        return random.choice([f for _, f in scored[:k]])

    def _pick_food_target_dual(self, game_state, my_pos, food_list):
        """
        Dual-offense target choice:
        - still likes close food for us,
        - but avoids region near teammate's pos/target,
        - if all food near teammate → return None (caller then returns home).
        """
        if not food_list:
            self.return_home(game_state)

        mate_info = self._get_offensive_teammate_info(game_state)
        if mate_info is None:
            # should not happen in dual-mode, but just in case:
            return self._pick_food_target_single(game_state, my_pos, food_list)

        _, mate_pos, mate_target = mate_info
        ref_pos = mate_target or mate_pos

        MIN_SEP = 3  # tiles considered "teammate's zone"

        # Filter food not too close to teammate region
        non_overlap = [
            f for f in food_list
            if self.get_maze_distance(f, ref_pos) >= MIN_SEP
        ]

        # If everything is in teammate’s pocket, let us go home instead
        if not non_overlap:
            return None

        # Prefer food close to us and far from teammate
        def cost(f):
            d_self = self.get_maze_distance(my_pos, f)
            d_mate = self.get_maze_distance(ref_pos, f)
            alpha = 0.6  # how strongly we avoid teammate region
            return d_self - alpha * d_mate

        return min(non_overlap, key=cost)


    def _safe_to_enter_trap(self, game_state, my_pos, food_pos):
        """Check if it's safe to enter a trap region for food."""
        if not self.topology.is_in_trap_region(food_pos):
            return True
            
        pocket = self.topology.pocket_id.get(food_pos)
        if pocket is None:
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
        return all(self.get_maze_distance(gpos, door) > required_dist 
                  for gpos in visible_ghosts)

    def _should_abort_trap(self, game_state, my_pos):
        """Check if we should abort being inside a trap."""
        if not self.topology.is_in_trap_region(my_pos):
            return False
            
        pocket = self.topology.pocket_id.get(my_pos)
        if pocket is None:
            return False
            
        exits = list(self.topology.pocket_exits[pocket])
            
        door = min(exits, key=lambda e: self.get_maze_distance(my_pos, e))
        pac_to_door = self.get_maze_distance(my_pos, door)
        
        visible_ghosts = self._get_visible_ghost_positions(game_state)
        if not visible_ghosts:
            return False
            
        ghost_to_door = min(self.get_maze_distance(g, door) for g in visible_ghosts)
        return ghost_to_door + 2 <= pac_to_door

    def hunt_food_single(self, game_state):
        return self._hunt_food_impl(game_state, dual=False)

    def hunt_food_dual(self, game_state):
        return self._hunt_food_impl(game_state, dual=True)

    def _hunt_food_impl(self, game_state, dual: bool):
        """Common food-hunting logic, with extra coordination when dual=True."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return Directions.STOP

        if self._should_abort_trap(game_state, my_pos):
            return self.return_home(game_state)

        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return self.return_home(game_state)

        # target selection differs for solo vs dual
        if dual:
            target = self._pick_food_target_dual(game_state, my_pos, food_list)
            # if we're on enemy side and we only have food in teammate's zone → go home
            if target is None and self.is_on_enemy_side(game_state):
                return self.return_home(game_state)
        else:
            target = self._pick_food_target_single(game_state, my_pos, food_list)

        if target is None:
            return Directions.STOP

        # remember our target so teammate can avoid our region
        BeliefBTOffensiveAgent.shared_targets[self.index] = target

        # Safety check for trap targets (same as before)
        if (target and self.topology.is_in_trap_region(target)
            and not self._safe_to_enter_trap(game_state, my_pos, target)):
            # Fall back to nearest food
            target = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))

        if target is None:
            return Directions.STOP

        path = self.pathfinder.find_path(
            game_state, my_pos, target, avoid_enemies=True
        )

        if path:
            return path[0]

        legal = game_state.get_legal_actions(self.index)
        return random.choice(legal) if legal else Directions.STOP

    def return_home(self, game_state):
        """Return to our side of the map."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return Directions.STOP
            
        home_tiles = self._get_home_tiles(game_state)
        if not home_tiles:
            return Directions.STOP
            
        nearest_home = min(home_tiles, key=lambda t: self.get_maze_distance(my_pos, t))
        path = self.pathfinder.find_path(game_state, my_pos, nearest_home, avoid_enemies=True)
        
        if path:
            return path[0]
            
        legal = game_state.get_legal_actions(self.index)
        return random.choice(legal) if legal else Directions.STOP

    def run_away(self, game_state):
        """Escape from ghosts."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return Directions.STOP
            
        # Check immediate danger
        close_ghosts = []
        immediate_danger = False
        
        for opp in self.get_opponents(game_state):
            st = game_state.get_agent_state(opp)
            pos = st.get_position()
            if pos is not None and not st.is_pacman:
                dist = self.get_maze_distance(my_pos, pos)
                if dist <= 1:
                    immediate_danger = True
                if dist <= 5:
                    close_ghosts.append(pos)
        
        # Safety override or probabilistic decision
        my_state = game_state.get_agent_state(self.index)
        p_home = min(1.0, float(my_state.num_carrying) / max(1, self.CARRY_THRESHOLD))
        
        if immediate_danger or len(close_ghosts) >= 2 or random.random() < p_home:
            return self._escape_via_best_option(game_state)
        
        # TODO : Make it SAFER!
        food_list = self.get_food(game_state).as_list()
        if food_list:
            # Pick random food weighted by distance (farther = higher weight)
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
        """Choose between home or power pellet for escape."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos is None:
            return Directions.STOP
            
        home_tiles = self._get_home_tiles(game_state)
        if not home_tiles:
            return Directions.STOP
            
        nearest_home = min(home_tiles, key=lambda t: self.get_maze_distance(my_pos, t))
        
        # If power active, just go home without avoiding
        if self.is_power_active(game_state):
            path, _ = self.pathfinder.find_path(
                game_state, my_pos, nearest_home, avoid_enemies=False, return_cost=True
            )
            if path:
                return path[0]
        else:
            # Compare home vs pellet options
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
                # No pellets, just go home
                path, _ = self.pathfinder.find_path(
                    game_state, my_pos, nearest_home, avoid_enemies=True, return_cost=True
                )
                if path:
                    return path[0]
        
        legal = game_state.get_legal_actions(self.index)
        return random.choice(legal) if legal else Directions.STOP
    
    ## 2 ATTACKERS 

    def _is_on_enemy_side_pos(self, game_state, pos):
        mid_x = game_state.get_walls().width // 2
        return pos[0] >= mid_x if self.red else pos[0] < mid_x    

    def _get_offensive_teammate_info(self, game_state):
        """
        Return (teammate_index, teammate_pos, teammate_target)
        if the teammate is currently in offensive mode.
        """
        for idx in self.team_indices:
            if idx == self.index:
                continue

            mode = BeliefBTOffensiveAgent.shared_modes.get(idx, None)
            if mode != "offense":
                continue

            st = game_state.get_agent_state(idx)
            if st is None or st.get_position() is None:
                continue

            pos = st.get_position()
            tgt = BeliefBTOffensiveAgent.shared_targets.get(idx, None)
            return idx, pos, tgt

        return None
