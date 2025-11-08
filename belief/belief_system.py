import contest.util as util
from contest.util import manhattan_distance

class GhostBeliefTracker:
    def __init__(self, agent, opponents):
        self.agent = agent
        self.opponents = opponents
        self.legal_positions = None
        self.beliefs = {opp: util.Counter() for opp in opponents}

    def initialize_uniformly(self, game_state):
        """Initialize beliefs to a uniform distribution over all legal tiles."""
        walls = game_state.get_walls()
        self.legal_positions = [
            (x, y)
            for x in range(walls.width)
            for y in range(walls.height)
            if not walls[x][y]
        ]
        for opp in self.opponents:
            self.beliefs[opp] = util.Counter()
            for p in self.legal_positions:
                self.beliefs[opp][p] = 1.0
            self.beliefs[opp].normalize()

    def observe(self, game_state):
        """
        Update beliefs given noisy distance observations and visible ghosts.
        """
        noisy_distances = game_state.get_agent_distances()
        my_pos = game_state.get_agent_position(self.agent.index)

        for opp in self.opponents:
            opp_pos = game_state.get_agent_position(opp)
            new_belief = util.Counter()

            if opp_pos is not None:
                # Visible ghost → collapse belief to a delta
                new_belief[opp_pos] = 1.0
            else:
                noisy_dist = noisy_distances[opp]
                for pos in self.legal_positions:
                    true_dist = manhattan_distance(my_pos, pos)
                    prob = game_state.get_distance_prob(true_dist, noisy_dist)
                    if prob > 0:
                        new_belief[pos] = prob * self.beliefs[opp][pos]
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
        walls = game_state.get_walls()
        new_beliefs = {}

        for opp in self.opponents:
            new_belief = util.Counter()
            for pos in self.legal_positions:
                x, y = pos
                possible_moves = [
                    (x + dx, y + dy)
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]
                    if not walls[x + dx][y + dy]
                ]
                if not possible_moves:
                    possible_moves = [pos]
                prob_per_move = self.beliefs[opp][pos] / len(possible_moves)
                for p2 in possible_moves:
                    new_belief[p2] += prob_per_move
            new_belief.normalize()
            new_beliefs[opp] = new_belief

        self.beliefs = new_beliefs
