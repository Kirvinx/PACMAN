import contest.util as util
from contest.util import manhattan_distance
from contest.capture import SONAR_NOISE_RANGE, SONAR_NOISE_VALUES

class GhostBeliefTracker:
    def __init__(self, agent, opponents):
        self.agent = agent
        self.opponents = opponents
        self.legal_positions = None
        self.legal_positions_set = None  # Optimization 3: Set for O(1) lookup
        self.neighbors = None  # Optimization 1: Pre-computed neighbors
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

        # --- legal positions ---
        self.legal_positions = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if not walls[x][y]
        ]
        self.legal_positions_set = set(self.legal_positions)

        # --- precompute neighbors for each legal position ---
        self.neighbors = {}
        for x, y in self.legal_positions:
            neighbors = [
                (x + dx, y + dy)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if 0 <= x + dx < width
                and 0 <= y + dy < height
                and not walls[x + dx][y + dy]
            ]
            # if dead-end, allow staying in place
            self.neighbors[(x, y)] = neighbors if neighbors else [(x, y)]

        # --- spawn-based initialization for each opponent ---
        for opp in self.opponents:
            spawn_pos = game_state.get_initial_agent_position(opp)
            self.beliefs[opp] = util.Counter()
            if spawn_pos is not None and spawn_pos in self.legal_positions_set:
                # delta at the true spawn
                self.beliefs[opp][spawn_pos] = 1.0
            else:
                # robust fallback: uniform over all legal positions
                uniform_prob = 1.0 / len(self.legal_positions)
                for p in self.legal_positions:
                    self.beliefs[opp][p] = uniform_prob


    def observe(self, game_state):
        """
        Update beliefs given noisy distance observations and visible ghosts.
        """
        noisy_distances = game_state.get_agent_distances()
        my_pos = game_state.get_agent_position(self.agent.index)

        for opp in self.opponents:
            opp_pos = game_state.get_agent_position(opp)

            if opp_pos is not None:
                # Visible ghost → delta belief
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

            # TODO : WHEN WE ARE SURE WE KILLED ENEMY; RESET BELIEF TO SPAWN ISNTEAD
            if new_belief.total_count() == 0:
                uniform_prob = 1.0 / len(self.legal_positions)
                new_belief = util.Counter({p: uniform_prob for p in self.legal_positions})

            self.beliefs[opp] = new_belief

        self._filter_side_constraints(game_state)
        eaten = self._detect_eaten_food(game_state)
        self._incorporate_eaten_food(eaten)

        self.draw_danger()


    def elapse_time(self, game_state):
        """
        Predict ghost motion — spread beliefs to neighboring legal positions.
        """
        new_beliefs = {}

        for opp in self.opponents:
            new_belief = util.Counter()
            
            # Optimization 2: Only iterate over positions with non-zero belief
            for pos, prob in self.beliefs[opp].items():
                if prob > 0:  # Skip zero probabilities
                    # Optimization 1: Use pre-computed neighbors
                    possible_moves = self.neighbors[pos]
                    prob_per_move = prob / len(possible_moves)
                    for p2 in possible_moves:
                        new_belief[p2] += prob_per_move
            
            new_belief.normalize()
            new_beliefs[opp] = new_belief

        self.beliefs = new_beliefs
        self.draw_danger()

    def draw_danger(self):
        # --- Choose which team may draw ---
        DRAW_FOR_RED = False
        DRAW_FOR_BLUE = True

        # Skip drawing if this agent's team is not allowed
        if self.agent.red and not DRAW_FOR_RED:
            return
        if not self.agent.red and not DRAW_FOR_BLUE:
            return

        # --- rare clearing ---
        self.debug_step += 1
        CLEAR_INTERVAL = 5   # clear every 5 calls

        # --- Aggregate danger ---
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
        color = [1.0, 0.0, 0.0]

        # --- Clear only sometimes ---
        if hasattr(self.agent, "debug_clear"):
            if self.debug_step % CLEAR_INTERVAL == 0:
                self.agent.debug_clear()

        # --- Draw every step ---
        if hasattr(self.agent, "debug_draw"):
            self.agent.debug_draw(cells, color, clear=False)

    def _filter_side_constraints(self, game_state):
        """
        Remove impossible belief mass based on whether the opponent is a Pacman
        (invading our side) or a Ghost (defending their side).

        This sharply reduces uncertainty.
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
                    if (self.agent.red and x < mid_x) or ((not self.agent.red) and x >= mid_x):
                        new_belief[(x, y)] = p
            else:
                # Opponent MUST be on THEIR side
                for (x, y), p in belief.items():
                    if (self.agent.red and x >= mid_x) or ((not self.agent.red) and x < mid_x):
                        new_belief[(x, y)] = p

            # If the filtered belief still has mass, use it
            if new_belief.total_count() > 0:
                new_belief.normalize()
                self.beliefs[opp] = new_belief
    
    

    def _detect_eaten_food(self, game_state):
        """
        Detect which food dots (on our side) disappeared since last turn.
        Uses the agent's own API (get_food_you_are_defending).
        """
        current_food = self.agent.get_food_you_are_defending(game_state)

        if not hasattr(self, "_previous_defended_food"):
            self._previous_defended_food = current_food
            return []

        old = self._previous_defended_food.as_list()
        new = current_food.as_list()

        eaten = [pos for pos in old if pos not in new]

        # Store for next time
        self._previous_defended_food = current_food

        return eaten
    
    def _opponent_could_have_eaten(self, opp, eaten_positions):
        """
        Returns True if this opponent's belief has support at or near
        any eaten food tile.
        """
        belief = self.beliefs[opp]
        if not belief:
            return False

        for ex, ey in eaten_positions:
            # Direct tile
            if belief[(ex, ey)] > 0:
                return True

            # Adjacent tiles allow timing offset
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = ex + dx, ey + dy
                if belief[(nx, ny)] > 0:
                    return True

        return False


    def _incorporate_eaten_food(self, eaten_positions):
        """
        Sharpen beliefs toward tiles where food disappeared,
        but only for opponents who could realistically be there.
        """
        if not eaten_positions:
            return

        for opp in self.opponents:

            # Skip enemies with no chance of eating this food
            if not self._opponent_could_have_eaten(opp, eaten_positions):
                continue

            belief = self.beliefs[opp]
            if not belief:
                continue

            new_belief = util.Counter()

            # Boost probability on tiles adjacent to eaten food
            for pos, p in belief.items():
                for (fx, fy) in eaten_positions:
                    if manhattan_distance(pos, (fx, fy)) <= 1:
                        new_belief[pos] += p

            # If we got any mass from this, normalize
            if new_belief.total_count() > 0:
                new_belief.normalize()
                self.beliefs[opp] = new_belief



