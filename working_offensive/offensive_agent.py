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

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        self.topology = MapTopologyAnalyzer(game_state.get_walls())
        self.belief_tracker = GhostBeliefTracker(self, self.get_opponents(game_state))
        self.belief_tracker.initialize_uniformly(game_state)
        self.pathfinder = AStarPathfinder(self)

        # Build Behavior Tree
        self.behavior_tree = Selector([
            # Emergency retreat if ghost nearby on enemy side
            Sequence([
                Condition(lambda agent, gs: 
                    agent.is_on_enemy_side(gs) 
                    and not agent.is_power_active(gs)
                    and agent.ghost_nearby(gs, radius=5)),
                Action(lambda agent, gs: agent.run_away(gs)),
            ]),
            # Return home if carrying too much food
            Sequence([
                Condition(lambda agent, gs: 
                    gs.get_agent_state(agent.index).num_carrying >= agent.CARRY_THRESHOLD),
                Action(lambda agent, gs: agent.return_home(gs)),
            ]),
            # Default: hunt food
            Action(lambda agent, gs: agent.hunt_food(gs)),
        ])

    def choose_action(self, game_state):
        # Update beliefs
        self.belief_tracker.elapse_time(game_state)
        self.belief_tracker.observe(game_state)

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

    def _pick_food_target(self, game_state, my_pos, food_list):
        """Choose food target based on current side."""
        if not food_list:
            return None
            
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

    def hunt_food(self, game_state):
        """Choose food target and move toward it."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        if self._should_abort_trap(game_state, my_pos):
            return self.return_home(game_state)
            
        food_list = self.get_food(game_state).as_list()
        if not food_list or my_pos is None:
            return Directions.STOP
            
        target = self._pick_food_target(game_state, my_pos, food_list)
        
        # Safety check for trap targets
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
        
        # Try to continue hunting (pick far food)
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