import contest.util as util
from contest.util import manhattan_distance
from contest.capture import SONAR_NOISE_RANGE, SONAR_NOISE_VALUES

class GhostBeliefTracker:
    def __init__(self, agent, opponents, team_indices):
        """
        Initialize a team-shared belief tracker.
        
        Args:
            agent: Any agent from the team (used for team color and debug/food APIs)
            opponents: List of opponent indices to track
        """
        # Team-level properties
        self.is_red = agent.red
        self.debug_agent = agent  # Used for debug_draw and get_food_you_are_defending
        self.opponents = opponents
        self.team_indices = team_indices

        # Belief tracking structures
        self.legal_positions = None
        self.legal_positions_set = None  # Set for O(1) lookup
        self.neighbors = None  # Pre-computed neighbors for efficiency
        self.beliefs = {opp: util.Counter() for opp in opponents}
        self.debug_step = 0

    def get_belief(self, opp_index):
        """
        Returns the belief Counter for a given opponent index.
        """
        return self.beliefs.get(opp_index, util.Counter())

    def initialize_uniformly(self, game_state):
        """
        Initialize beliefs for each opponent.
        
        Uses:
        - legal_positions / neighbors precomputation (for later updates)
        - get_initial_agent_position(opp) to set a delta belief at each opp's spawn
        """
        walls = game_state.get_walls()
        width, height = walls.width, walls.height

        # --- Precompute legal positions ---
        self.legal_positions = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if not walls[x][y]
        ]
        self.legal_positions_set = set(self.legal_positions)

        # --- Precompute neighbors for each legal position ---
        self.neighbors = {}
        for x, y in self.legal_positions:
            neighbors = [
                (x + dx, y + dy)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if 0 <= x + dx < width
                and 0 <= y + dy < height
                and not walls[x + dx][y + dy]
            ]
            # If dead-end, allow staying in place
            self.neighbors[(x, y)] = neighbors if neighbors else [(x, y)]

        # --- Spawn-based initialization for each opponent ---
        for opp in self.opponents:
            spawn_pos = game_state.get_initial_agent_position(opp)
            self.beliefs[opp] = util.Counter()
            if spawn_pos is not None and spawn_pos in self.legal_positions_set:
                # Delta function at the true spawn position
                self.beliefs[opp][spawn_pos] = 1.0
            else:
                # Robust fallback: uniform over all legal positions
                uniform_prob = 1.0 / len(self.legal_positions)
                for p in self.legal_positions:
                    self.beliefs[opp][p] = uniform_prob

    def observe(self, game_state, observing_index, is_leader):
        """
        Update beliefs given noisy distance observations and visible ghosts,
        using the sonar reading from the agent with index observing_index.
        
        Args:
            game_state: Current game state
            observing_index: Index of the agent whose sonar we're using
        """
        noisy_distances = game_state.get_agent_distances()
        my_pos = game_state.get_agent_position(observing_index)

        for opp in self.opponents:
            opp_pos = game_state.get_agent_position(opp)

            if opp_pos is not None:
                # Visible ghost → delta belief at exact position
                self.beliefs[opp] = util.Counter()
                self.beliefs[opp][opp_pos] = 1.0
                continue

            new_belief = util.Counter()
            noisy_dist = noisy_distances[opp]

            # Iterate only over positions with positive belief
            for pos, prior in self.beliefs[opp].items():
                if prior <= 0:
                    continue

                true_dist = manhattan_distance(my_pos, pos)
                noise = noisy_dist - true_dist

                # --- Corrected noise model ---
                if noise in SONAR_NOISE_VALUES:
                    prob = 1.0 / SONAR_NOISE_RANGE

                # ±7 spillover case (rare float/integer rounding artifact)
                elif (noise == SONAR_NOISE_VALUES[0] - 1 or
                      noise == SONAR_NOISE_VALUES[-1] + 1):
                    prob = (1.0 / SONAR_NOISE_RANGE) * 0.2

                else:
                    prob = 0.0

                new_belief[pos] = prob * prior

            new_belief.normalize()

            # If belief collapsed (e.g., killed enemy), reset to spawn
            if new_belief.total_count() == 0:
                uniform_prob = 1.0 / len(self.legal_positions)
                new_belief = util.Counter({p: uniform_prob for p in self.legal_positions})

            self.beliefs[opp] = new_belief

        # Apply additional constraints
        if is_leader:
            self._filter_side_constraints(game_state)
            eaten = self._detect_eaten_food(game_state)
            self._incorporate_eaten_food(eaten)


        #self.draw_danger()

    def elapse_time(self, game_state):
        """
        Predict ghost motion — spread beliefs to neighboring legal positions.
        """
        new_beliefs = {}

        for opp in self.opponents:
            new_belief = util.Counter()
            
            # Only iterate over positions with non-zero belief (optimization)
            for pos, prob in self.beliefs[opp].items():
                if prob > 0:  # Skip zero probabilities
                    # Use pre-computed neighbors
                    possible_moves = self.neighbors[pos]
                    prob_per_move = prob / len(possible_moves)
                    for p2 in possible_moves:
                        new_belief[p2] += prob_per_move
            
            new_belief.normalize()
            new_beliefs[opp] = new_belief

        self.beliefs = new_beliefs
        #self.draw_danger()

    def draw_danger(self):
        """
        Draw the danger map showing belief distributions.
        Uses team color to determine whether to draw.
        """
        # --- Choose which team may draw ---
        DRAW_FOR_RED = False
        DRAW_FOR_BLUE = True

        # No debug agent → nothing to draw
        if self.debug_agent is None:
            return

        # Skip drawing if this agent's team is not allowed
        if self.is_red and not DRAW_FOR_RED:
            return
        if (not self.is_red) and not DRAW_FOR_BLUE:
            return

        # --- Rare clearing to prevent clutter ---
        self.debug_step += 1
        CLEAR_INTERVAL = 5  # Clear every 5 calls

        # --- Aggregate danger from all opponents ---
        danger = util.Counter()
        for opp in self.opponents:
            for pos, p in self.beliefs[opp].items():
                danger[pos] += p

        if not danger:
            return

        max_p = max(danger.values())
        if max_p == 0:
            return

        # Draw all tiles with probability > 0
        cells = [pos for pos, p in danger.items() if p > 0]
        color = [1.0, 0.0, 0.0]  # Red for danger

        # --- Clear only sometimes ---
        if hasattr(self.debug_agent, "debug_clear"):
            if self.debug_step % CLEAR_INTERVAL == 0:
                self.debug_agent.debug_clear()

        # --- Draw every step ---
        if hasattr(self.debug_agent, "debug_draw"):
            self.debug_agent.debug_draw(cells, color, clear=False)

    def _filter_side_constraints(self, game_state):
        """
        Remove impossible belief mass based on whether the opponent is a Pacman
        (invading our side) or a Ghost (defending their side).
        
        This sharply reduces uncertainty by using game rules.
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2

        for opp in self.opponents:
            state = game_state.get_agent_state(opp)
            belief = self.beliefs[opp]

            if not belief:
                continue

            new_belief = util.Counter()

            if state.is_pacman:
                # Opponent MUST be on OUR side
                for (x, y), p in belief.items():
                    if (self.is_red and x < mid_x) or ((not self.is_red) and x >= mid_x):
                        new_belief[(x, y)] = p
            else:
                # Opponent MUST be on THEIR side
                for (x, y), p in belief.items():
                    if (self.is_red and x >= mid_x) or ((not self.is_red) and x < mid_x):
                        new_belief[(x, y)] = p

            # If the filtered belief still has mass, use it
            if new_belief.total_count() > 0:
                new_belief.normalize()
                self.beliefs[opp] = new_belief

    def _detect_eaten_food(self, game_state):
        """
        Detect which food dots (on our side) disappeared since last turn.
        Uses ANY one agent on our team (debug_agent) to query defended food.
        """
        if self.debug_agent is None:
            return []

        current_food = self.debug_agent.get_food_you_are_defending(game_state)

        if not hasattr(self, "_previous_defended_food"):
            self._previous_defended_food = current_food
            return []

        old = self._previous_defended_food.as_list()
        new = current_food.as_list()

        eaten = [pos for pos in old if pos not in new]

        # Store for next time
        self._previous_defended_food = current_food

        return eaten

    def _food_eating_likelihood(self, opp, eaten_pos):
        """
        Returns a score indicating how likely it is that this opponent
        ate the food at eaten_pos.
        Higher = more likely.
        """
        belief = self.beliefs[opp]
        if not belief:
            return 0.0

        (fx, fy) = eaten_pos
        total = 0.0

        # Consider all tiles within manhattan distance 1
        for dx, dy in [(0,0), (1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = fx + dx, fy + dy
            total += belief[(nx, ny)]

        return total


    def _incorporate_eaten_food(self, eaten_positions):
        """
        Assign eaten food to the *most likely* opponent.
        Collapse only that opponent's belief.
        """
        if not eaten_positions:
            return

        for eaten_pos in eaten_positions:

            # --- Score each opponent ---
            scores = {}
            for opp in self.opponents:
                scores[opp] = self._food_eating_likelihood(opp, eaten_pos)

            # --- Pick the maximally likely eater ---
            best_opp = max(scores, key=lambda o: scores[o])

            # If nobody has any mass near the eaten tile, skip
            if scores[best_opp] == 0:
                continue

            # --- Collapse belief for ONLY the best opponent ---
            belief = self.beliefs[best_opp]
            new_belief = util.Counter()

            # Keep probability only for tiles near the eaten food (distance ≤ 1)
            (fx, fy) = eaten_pos
            for pos, p in belief.items():
                if manhattan_distance(pos, (fx, fy)) <= 1:
                    new_belief[pos] += p

            # Normalize
            if new_belief.total_count() > 0:
                new_belief.normalize()
                self.beliefs[best_opp] = new_belief
