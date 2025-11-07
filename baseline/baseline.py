from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import manhattan_distance, nearest_point
from contest.agents.team_name_1.baseline.offensive_agent import OffensiveAgent
from contest.agents.team_name_1.baseline.behavior_tree import Selector, Sequence, Condition, Action
from contest.agents.team_name_1.baseline.offensive_agent import OffensiveAgent
import random
import contest.util as util

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent'):
    """
    Called by capture.py to create your two agents.
    """
    return [OffensiveAgent(first_index), OffensiveAgent(second_index)]

