import contest.util as util
from contest.util import manhattan_distance

class GhostBeliefTracker:
    def __init__(self, agent, opponents):
        self.agent = agent
        self.opponents = opponents
        self.legal_positions = None
        self.legal_positions_set = None  # Optimization 3: Set for O(1) lookup
        self.neighbors = None  # Optimization 1: Pre-computed neighbors
        self.beliefs = {opp: util.Counter() for opp in opponents}

    def get_belief(self, opp_index):
        """
        Returns the belief Counter for a given opponent index.
        """
        return self.beliefs.get(opp_index, util.Counter())

    def initialize_uniformly(self, game_state):
        """Initialize beliefs to a uniform distribution over all legal tiles."""
        walls = game_state.get_walls()
        self.legal_positions = [
            (x, y)
            for x in range(walls.width)
            for y in range(walls.height)
            if not walls[x][y]
        ]
        
        # Optimization 3: Convert to set for O(1) lookup
        self.legal_positions_set = set(self.legal_positions)
        
        # Optimization 1: Pre-compute neighbors for each position
        self.neighbors = {}
        for x, y in self.legal_positions:
            neighbors = [
                (x + dx, y + dy)
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]
                if 0 <= x + dx < walls.width and 0 <= y + dy < walls.height 
                   and not walls[x + dx][y + dy]
            ]
            self.neighbors[(x, y)] = neighbors if neighbors else [(x, y)]
        
        # Initialize beliefs uniformly
        uniform_prob = 1.0 / len(self.legal_positions)
        for opp in self.opponents:
            self.beliefs[opp] = util.Counter()
            for p in self.legal_positions:
                self.beliefs[opp][p] = uniform_prob
            # Already normalized by using uniform_prob

    def observe(self, game_state):
        """
        Update beliefs given noisy distance observations and visible ghosts.
        """
        noisy_distances = game_state.get_agent_distances()
        my_pos = game_state.get_agent_position(self.agent.index)

        for opp in self.opponents:
            opp_pos = game_state.get_agent_position(opp)

            if opp_pos is not None:
                # Visible ghost → collapse belief to a delta
                self.beliefs[opp] = util.Counter()
                self.beliefs[opp][opp_pos] = 1.0
            else:
                new_belief = util.Counter()
                noisy_dist = noisy_distances[opp]
                
                # Optimization 2: Only iterate over positions with non-zero belief
                for pos, prior in self.beliefs[opp].items():
                    if prior > 0:  # Skip zero probabilities
                        true_dist = manhattan_distance(my_pos, pos)
                        prob = game_state.get_distance_prob(true_dist, noisy_dist)
                        if prob > 0:
                            new_belief[pos] = prob * prior
                
                new_belief.normalize()
                if new_belief.total_count() == 0:
                    # Reinitialize if no possible positions (happens after teleports)
                    self.initialize_uniformly(game_state)
                else:
                    self.beliefs[opp] = new_belief

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