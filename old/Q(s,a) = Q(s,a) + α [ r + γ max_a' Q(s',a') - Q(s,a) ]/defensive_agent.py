from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import manhattan_distance
from contest.agents.team_name_1.beliefline.topology import MapTopologyAnalyzer
from contest.agents.team_name_1.beliefline.belief_system_opt import GhostBeliefTracker
import random
import numpy as np
import os


ALL_ACTIONS = [
    Directions.NORTH,
    Directions.SOUTH,
    Directions.EAST,
    Directions.WEST,
    Directions.STOP,
]


class BeliefRLDefensiveAgent(CaptureAgent):
    """
    Defensive agent using Approximate Q-Learning:

        Q(s, a) = w_a · φ(s, a)

    Features φ(s, a) are computed from the successor state s' = generate_successor(s, a),
    and incorporate:
      - invader distances (visible + belief)
      - topology (trap depth, articulation)
      - pellet / entry proximity
      - teammate coordination
    """

    def __init__(self, index, num_training=0, **kwargs):
        super().__init__(index)

        # how many episodes should be training (from -x)
        self.num_training_episodes = int(num_training)
        self.episodes_so_far = 0

        # RL hyperparams
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.episode_reward_sum = 0.0

        # Debug
        self.debug_td_sum = 0.0
        self.debug_td_count = 0

        # will be set in register_initial_state
        self.topology = None
        self.belief_tracker = None
        self.entry_points = []

        # Q-learning
        self.weights = None      # init lazily in register_initial_state
        self.training = self.num_training_episodes > 0

        # traces
        self.last_state = None
        self.last_action = None

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # --- Map & belief setup ---
        self.topology = MapTopologyAnalyzer(game_state.get_walls())
        opponents = self.get_opponents(game_state)
        self.belief_tracker = GhostBeliefTracker(self, opponents)
        self.belief_tracker.initialize_uniformly(game_state)
        food_grid = game_state.get_red_food() if game_state.is_on_red_team(self.index) else game_state.get_blue_food()
        self.initial_defended_food_count = len(food_grid.as_list())

        # Precompute entry points along the mid border on *our* side
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2
        my_red = game_state.is_on_red_team(self.index)
        border_x = mid_x - 1 if my_red else mid_x
        self.entry_points = [
            (border_x, y)
            for y in range(height)
            if not walls[border_x][y]
        ]

        # new episode: clear traces
        self.last_state = None
        self.last_action = None
        self.episode_reward_sum = 0.0

        # this episode is training iff we’re still within num_training_episodes
        self.training = self.episodes_so_far < self.num_training_episodes

        # --- Initialize weights ONCE only ---
        if self.weights is None:
            # try to load checkpoint if it exists
            if os.path.exists("rl_defender_weights.npy"):
                try:
                    self.load_weights("rl_defender_weights.npy")
                    print("[INFO] Loaded defender weights from checkpoint ✅")
                except Exception as e:
                    print(f"[WARN] Could not load pretrained weights: {e}")
                    dummy_features = self._extract_features_sa(game_state, Directions.STOP)
                    f_dim = len(dummy_features)
                    self.weights = {a: np.random.randn(f_dim) * 0.01 for a in ALL_ACTIONS}
            else:
                dummy_features = self._extract_features_sa(game_state, Directions.STOP)
                f_dim = len(dummy_features)
                self.weights = {a: np.random.randn(f_dim) * 0.01 for a in ALL_ACTIONS}
    # ------------------------------------------------------------------
    # Q(s,a) and action selection
    # ------------------------------------------------------------------

    def get_q_value(self, game_state, action):
        """
        Q(s,a) = w_a · φ(s,a), but ONLY for legal actions.
        Illegal actions get a very low Q so they are never chosen.
        """
        legal = game_state.get_legal_actions(self.index)
        if action not in legal:
            return -1e6  # effectively -∞

        feat = self._extract_features_sa(game_state, action)
        return float(np.dot(self.weights[action], feat))
    
    def get_action_values(self, game_state):
        """
        Return Q(s,a) for all actions, but only compute for legal ones.
        Illegal ones are filled with a large negative number.
        """
        legal = game_state.get_legal_actions(self.index)
        q_values = np.full(len(ALL_ACTIONS), -1e6, dtype=float)

        for a in legal:
            idx = ALL_ACTIONS.index(a)
            q_values[idx] = self.get_q_value(game_state, a)

        return q_values

    def get_action(self, game_state):
        """
        Called by the engine every turn.

        We:
          1) If training and we have a previous (s,a), perform a Q-update
             using reward from (last_state -> game_state).
          2) Call the normal CaptureAgent.get_action, which calls choose_action.
        """
        # this will append to observation_history and call self.choose_action(...)
        return super().get_action(game_state)

    def choose_action(self, game_state):
        """
        ε-greedy over Q(s,a) on *legal* actions.
        Store (s,a) for later Q-learning update when self.training is True.
        """

        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP

        # Epsilon decay in training
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # ε-greedy
        if self.training and random.random() < self.epsilon:
            chosen = random.choice(legal)
        else:
            q_values = self.get_action_values(game_state)  # already only legal are high
            best_idx = int(q_values.argmax())
            chosen = ALL_ACTIONS[best_idx]

        # Remember (s,a) for update
        if self.training:
            self.last_state = game_state  # deep_copy from engine
            self.last_action = chosen

        return chosen

    def _extract_features_sa(self, game_state, action):
        """
        Features for (state, action).
        We assume 'action' is legal (get_q_value guards this), but we can still
        be defensive and short-circuit if needed.
        """
        if action not in game_state.get_legal_actions(self.index):
            # shouldn't happen if get_q_value is used correctly
            # return a zero vector of the right size to be safe
            return np.zeros_like(next(iter(self.weights.values())))

        successor = game_state.generate_successor(self.index, action)
        return self._features_from_successor(game_state, successor, action)


    # ------------------------------------------------------------------
    # Q-learning update (to be called by external training harness)
    # ------------------------------------------------------------------

    def update(self, next_state, reward):
        """
        Perform one Q-learning update using stored (s, a) and (s', r).

        Call pattern (in training harness):
          - before step: a = agent.choose_action(s)
          - env applies action → s'
          - r = agent.compute_reward(s, s')
          - agent.update(s', r)
        """
        if self.last_state is None or self.last_action is None:
            return

        # 1. φ(s,a) and current Q(s,a)
        phi_sa = self._extract_features_sa(self.last_state, self.last_action)
        q_old = float(np.dot(self.weights[self.last_action], phi_sa))

        # 2. max_{a'} Q(s', a') over legal actions in next_state
        legal_next = next_state.get_legal_actions(self.index)
        if legal_next:
            q_next_vals = [self.get_q_value(next_state, a_next) for a_next in legal_next]
            max_q_next = max(q_next_vals)
        else:
            max_q_next = 0.0

        # 3. TD target & error
        target = reward + self.gamma * max_q_next
        td_error = target - q_old

        # --- debug: accumulate TD error stats ---
        self.debug_td_sum += abs(td_error)
        self.debug_td_count += 1

        # 4. Gradient step on weights of chosen action
        self.weights[self.last_action] += self.alpha * td_error * phi_sa

        # 5. Clear trace
        self.last_state = None
        self.last_action = None


    # ------------------------------------------------------------------
    # Feature extraction φ(s,a) via successor
    # ------------------------------------------------------------------

    def _features_from_successor(self, prev_state, succ_state, action):
        """
        Successor-based features for defense (reduced set).

        Core ideas:
        - where invaders are (or are believed to be)
        - how far we and our teammate are from them / border / entry points
        - how much food we still defend
        - whether we're scared
        - simple blocking flag
        """
        my_state = succ_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        if my_pos is None:
            my_pos = (0, 0)

        walls = succ_state.get_walls()
        width, height = walls.width, walls.height
        max_geom_dist = float(width + height)

        my_red = succ_state.is_on_red_team(self.index)
        enemies = self.get_opponents(succ_state)

        # --------------------------------------------------------------
        # Visible invaders & enemy carrying
        # --------------------------------------------------------------
        invader_positions_visible = []
        enemy_carrying_sum = 0

        for e in enemies:
            st = succ_state.get_agent_state(e)
            if st is None:
                continue
            enemy_carrying_sum += st.num_carrying

            pos_e = st.get_position()
            if st.is_pacman and pos_e is not None:
                invader_positions_visible.append(pos_e)

        enemy_carrying_sum_norm = enemy_carrying_sum / 20.0  # rough normalization

        # --------------------------------------------------------------
        # Belief-based invader positions (if not visible)
        # --------------------------------------------------------------
        invader_positions_belief = []
        if hasattr(self, "belief_tracker"):
            for e in enemies:
                st = succ_state.get_agent_state(e)
                # if visible pacman, we already used it
                if st is not None and st.is_pacman and st.get_position() is not None:
                    continue
                belief = self.belief_tracker.get_belief(e)
                if belief:
                    best_pos = max(belief.items(), key=lambda kv: kv[1])[0]
                    invader_positions_belief.append(best_pos)

        # Use visible if we have them; else fall back to beliefs
        invader_refs = invader_positions_visible or invader_positions_belief

        # Number of visible invaders (on our side)
        num_invaders_norm = len(invader_positions_visible) / 2.0  # up to 2

        # Distance from us to nearest invader
        if invader_refs:
            d_self_inv = min(self.get_maze_distance(my_pos, p) for p in invader_refs)
            self_to_invader_norm = d_self_inv / max_geom_dist
        else:
            self_to_invader_norm = 1.0

        # --------------------------------------------------------------
        # Teammate information
        # --------------------------------------------------------------
        mate_state = None
        mate_pos = None
        for t in self.get_team(succ_state):
            if t != self.index:
                mate_state = succ_state.get_agent_state(t)
                if mate_state:
                    mate_pos = mate_state.get_position()
                break

        if mate_pos is not None and invader_refs:
            d_mate_inv = min(self.get_maze_distance(mate_pos, p) for p in invader_refs)
            mate_to_invader_norm = d_mate_inv / max_geom_dist
        else:
            mate_to_invader_norm = 1.0

        mate_defending = 0.0
        if mate_state is not None:
            mate_defending = 1.0 - float(mate_state.is_pacman)  # 1 if ghost, 0 if Pacman

        # --------------------------------------------------------------
        # Border / home distances
        # --------------------------------------------------------------
        mid_x = width // 2
        border_x = mid_x - 1 if my_red else mid_x
        border_positions = [
            (border_x, yy)
            for yy in range(height)
            if not walls[border_x][yy]
        ]

        if border_positions:
            d_self_border = min(self.get_maze_distance(my_pos, b) for b in border_positions)
            self_to_border_norm = d_self_border / max_geom_dist
        else:
            self_to_border_norm = 0.0

        if invader_refs and border_positions:
            d_inv_border = min(
                self.get_maze_distance(inv_pos, b)
                for inv_pos in invader_refs
                for b in border_positions
            )
            invader_to_border_norm = d_inv_border / max_geom_dist
        else:
            invader_to_border_norm = 1.0

        # --------------------------------------------------------------
        # Defended food
        # --------------------------------------------------------------
        current_food = (
            succ_state.get_red_food() if my_red else succ_state.get_blue_food()
        ).as_list()

        if getattr(self, "initial_defended_food_count", 0) > 0:
            defended_food_norm = len(current_food) / float(self.initial_defended_food_count)
        else:
            defended_food_norm = 0.0

        # --------------------------------------------------------------
        # Entry point distances
        # --------------------------------------------------------------
        inv_ref_pos = None
        if invader_refs:
            inv_ref_pos = min(invader_refs, key=lambda p: self.get_maze_distance(my_pos, p))

        if getattr(self, "entry_points", None) and my_pos is not None:
            d_entry_self = min(self.get_maze_distance(my_pos, e) for e in self.entry_points)
        else:
            d_entry_self = max_geom_dist
        d_entry_self_norm = d_entry_self / max_geom_dist

        if getattr(self, "entry_points", None) and inv_ref_pos is not None:
            d_entry_inv = min(self.get_maze_distance(inv_ref_pos, e) for e in self.entry_points)
        else:
            d_entry_inv = max_geom_dist
        d_entry_inv_norm = d_entry_inv / max_geom_dist

        # --------------------------------------------------------------
        # Blocking feature: am I between invader and border (roughly)?
        # --------------------------------------------------------------
        x, y = my_pos
        if inv_ref_pos is not None:
            inv_x, _ = inv_ref_pos
            if my_red:
                between = (inv_x <= x <= border_x) and (inv_x <= border_x)
            else:
                between = (border_x <= x <= inv_x) and (border_x <= inv_x)
            blocking_flag = 1.0 if between else 0.0
        else:
            blocking_flag = 0.0

        # --------------------------------------------------------------
        # Local openness (branching factor)
        # --------------------------------------------------------------
        legal_here = succ_state.get_legal_actions(self.index)
        open_ratio = len(legal_here) / 4.0

        # --------------------------------------------------------------
        # Scared timer
        # --------------------------------------------------------------
        self_scared_norm = my_state.scared_timer / 40.0

        # --------------------------------------------------------------
        # Final feature vector (reduced)
        # --------------------------------------------------------------
        feat = np.array([
            num_invaders_norm,         # 0
            self_to_invader_norm,      # 1
            mate_to_invader_norm,      # 2
            invader_to_border_norm,    # 3
            self_to_border_norm,       # 4
            defended_food_norm,        # 5
            enemy_carrying_sum_norm,   # 6
            d_entry_self_norm,         # 7
            d_entry_inv_norm,          # 8
            blocking_flag,             # 9
            open_ratio,                # 10
            self_scared_norm,          # 11
            mate_defending,            # 12
        ], dtype=float)

        return feat




    # ------------------------------------------------------------------
    # Reward shaping helper
    # ------------------------------------------------------------------

    def _did_catch_invader(self, prev_state, curr_state, enemy_idx):
        """
        Returns True if this agent likely caught the given enemy.

        Conditions:
            1. Previously: enemy was Pac-Man and within ≤2 tiles of us.
            2. Now: enemy is a Ghost (not Pac-Man).
            3. Enemy's current position is either:
                - invisible (respawned), OR
                - visible and near their spawn point.
            4. We moved toward the enemy (or stayed equally close),
                not away from them.
        """

        st_prev = prev_state.get_agent_state(enemy_idx)
        st_curr = curr_state.get_agent_state(enemy_idx)

        if not st_prev or not st_curr:
            return False

        # Must have been Pac-Man before, now Ghost
        if not st_prev.is_pacman or st_curr.is_pacman:
            return False

        # Our positions
        my_prev_pos = prev_state.get_agent_state(self.index).get_position()
        my_curr_pos = curr_state.get_agent_state(self.index).get_position()
        if my_prev_pos is None or my_curr_pos is None:
            return False

        # Enemy position before (needed for proximity)
        enemy_prev_pos = prev_state.get_agent_position(enemy_idx)
        if enemy_prev_pos is None:
            return False

        # close enough before transition (≤2 tiles apart)
        d_prev = self.get_maze_distance(my_prev_pos, enemy_prev_pos)
        if d_prev > 2:
            return False

        # must have moved toward enemy (or stayed equally close)
        d_curr = self.get_maze_distance(my_curr_pos, enemy_prev_pos)
        if d_curr > d_prev:
            return False  # moved away

        # now enemy is a ghost — check respawn plausibility
        enemy_curr_pos = curr_state.get_agent_position(enemy_idx)
        spawn_pos = curr_state.get_initial_agent_position(enemy_idx)

        # either invisible (means respawned) or visible & near spawn
        if enemy_curr_pos is None:
            return True  # out of vision → likely in respawn
        if spawn_pos and self.get_maze_distance(enemy_curr_pos, spawn_pos) <= 2:
            return True  # visible and near spawn → just respawned

        return False


    def compute_reward(self, prev_state, curr_state):
        """
        Reward shaping for defensive Q-learning.

        - +5 for catching an invader (Pacman -> Ghost, without them scoring)
        - -1 per defended food lost
        - small early-game bonus for moving out of spawn
        - small bonus for moving toward believed enemy positions
        - bonus when we newly make an invader visible and are within vision radius
        """

        reward = 0.0

        enemies = self.get_opponents(curr_state)
        my_red = curr_state.is_on_red_team(self.index)

        # --------------------------------------------------
        # 1) Catch reward (Pacman -> Ghost)
        # --------------------------------------------------
        for e in enemies:
            if self._did_catch_invader(prev_state, curr_state, e):
                reward += 5.0

        # --------------------------------------------------
        # 2) Food protection: penalize defended food loss
        # --------------------------------------------------
        prev_food = (prev_state.get_red_food() if my_red else prev_state.get_blue_food()).as_list()
        curr_food = (curr_state.get_red_food() if my_red else curr_state.get_blue_food()).as_list()
        lost = len(prev_food) - len(curr_food)
        if lost > 0:
            reward -= 1.0 * lost

        # --------------------------------------------------
        # Grab our positions
        # --------------------------------------------------
        my_prev = prev_state.get_agent_state(self.index).get_position()
        my_curr = curr_state.get_agent_state(self.index).get_position()

        # --------------------------------------------------
        # 3) Early-game: encourage moving out of spawn area
        # --------------------------------------------------
        time_left = getattr(curr_state.data, "timeleft", 1200)
        total_time = 1200.0
        progress = 1.0 - (time_left / total_time)

        if (
            progress <= 0.1 and  # only first ~10% of game
            my_prev is not None and
            my_curr is not None and
            hasattr(self, "start_pos") and
            self.start_pos is not None
        ):
            d_prev = self.get_maze_distance(self.start_pos, my_prev)
            d_curr = self.get_maze_distance(self.start_pos, my_curr)

            if d_curr > d_prev:
                # reward for moving further from spawn
                reward += 0.05 * (d_curr - d_prev)
            # no "reward = 0.0" here – we just stop giving this bonus once far enough

        # --------------------------------------------------
        # 4) Belief-based chasing: move toward most likely invader
        # --------------------------------------------------
        if (
            my_prev is not None and
            my_curr is not None and
            hasattr(self, "belief_tracker") and
            self.belief_tracker is not None
        ):
            belief_positions = []
            for e in enemies:
                belief = self.belief_tracker.get_belief(e)
                if belief and belief.total_count() > 0:
                    # most probable location of this opponent
                    best_pos = max(belief.items(), key=lambda kv: kv[1])[0]
                    belief_positions.append(best_pos)

            if belief_positions:
                # use current belief for both distances (approx): how much closer did we move
                d_prev_belief = min(self.get_maze_distance(my_prev, p) for p in belief_positions)
                d_curr_belief = min(self.get_maze_distance(my_curr, p) for p in belief_positions)

                improvement = d_prev_belief - d_curr_belief  # >0 => moved closer
                if improvement > 0:
                    reward += 0.1 * improvement  # gentle bonus, no penalty if < 0


        # --------------------------------------------------
        # 5) Penalty for enemy getting food over the border
        # --------------------------------------------------

        score_diff = curr_state.get_score() - prev_state.get_score()
        # enemy_score_gain: positive only if the *opponent* increased their score
        if curr_state.is_on_red_team(self.index):
            # we are red -> enemy is blue; enemy gain corresponds to negative global diff
            enemy_score_gain = max(0.0, -score_diff)
        else:
            # we are blue -> enemy is red; enemy gain corresponds to positive global diff
            enemy_score_gain = max(0.0, score_diff)

        # apply penalty proportional to enemy gain (tune multiplier as needed)
        ENEMY_SCORE_PENALTY = 2
        if enemy_score_gain > 0:
            reward -= ENEMY_SCORE_PENALTY * enemy_score_gain

        # --------------------------------------------------
        # 6) Penalty for going to the enemy side 
        # --------------------------------------------------
        if my_curr is not None:
            walls = curr_state.get_walls()
            width = walls.width
            mid_x = width // 2
            my_red = curr_state.is_on_red_team(self.index)

            # For red, our side is x < mid_x; for blue, our side is x >= mid_x
            on_enemy_side = (my_red and my_curr[0] >= mid_x) or (not my_red and my_curr[0] < mid_x)

            if on_enemy_side:
                # big penalty for leaving our defensive half
                reward -= 3.0

        return reward


    
    def observation_function(self, game_state):
        """
        Called by the game engine before each move.
        We:
        1) update belief tracker,
        2) do Q-learning update from (last_state, last_action) -> game_state.
        """
        # update beliefs for new observation
        if self.belief_tracker is not None:
            self.belief_tracker.elapse_time(game_state)
            self.belief_tracker.observe(game_state)

        # Q-learning update from previous transition
        if self.training and self.last_state is not None and self.last_action is not None:
            reward = self.compute_reward(self.last_state, game_state)
            self.update(game_state, reward)   # will clear last_state/last_action
            self.episode_reward_sum += reward


        # remember this state for the NEXT transition
        self.last_state = game_state
        return game_state
    
    def final(self, game_state):
        if self.training and self.last_state is not None and self.last_action is not None:
            reward = self.compute_reward(self.last_state, game_state)
            self.update(game_state, reward)

        self.last_state = None
        self.last_action = None

        self.episodes_so_far += 1

        final_score = game_state.get_score()
        print(f"[TRAIN] Episode {self.episodes_so_far}/{self.num_training_episodes} "
      f"| Env score: {final_score:.2f} | RL total reward: {self.episode_reward_sum:.2f}")

        with open("rl_defender_training_log.txt", "a") as f:
            f.write(f"{self.episodes_so_far},{final_score}\n")

        # --- debug: TD error + weight norms ---
        if self.debug_td_count > 0:
            avg_td = self.debug_td_sum / float(self.debug_td_count)
        else:
            avg_td = 0.0
        print(f"[DBG] Avg |TD error| this episode: {avg_td:.4f}")
        for a in ALL_ACTIONS:
            w_norm = np.linalg.norm(self.weights[a])
            print(f"[DBG] ||w[{a}]|| = {w_norm:.4f}")

        # reset debug accumulators for next episode
        self.debug_td_sum = 0.0
        self.debug_td_count = 0

        # periodic checkpoint every N episodes (e.g. 50)
        if self.episodes_so_far % 50 == 0:
            self.save_weights("rl_defender_weights.npy")
            print("[TRAIN] Checkpoint saved ✅")

        if self.episodes_so_far >= self.num_training_episodes:
            self.training = False
            self.save_weights("rl_defender_weights.npy")
            print("[TRAIN] Training finished. Weights saved to rl_defender_weights.npy ✅")



    # ------------------------------------------------------------------
    # Save / load helpers for weights
    # ------------------------------------------------------------------

    def save_weights(self, path):
        data = {a: w.tolist() for a, w in self.weights.items()}
        np.save(path, data)

    def load_weights(self, path):
        data = np.load(path, allow_pickle=True).item()
        self.weights = {a: np.array(v, dtype=float) for a, v in data.items()}
