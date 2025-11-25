import random
import contest.util as util

from contest.agents.team_name_1.version2.offensive_agent2 import BeliefBTOffensiveAgent2
from contest.agents.team_name_1.version2.offensive_agent import BeliefBTOffensiveAgent
from contest.agents.team_name_1.version2.defensive_agent import BeliefBTDefensiveAgent
from contest.agents.team_name_1.version2.defensive_agent2 import BeliefBTDefensiveAgent2
from contest.game import Directions
from contest.util import nearest_point
from contest.baselineTeam import DefensiveReflexAgent
from contest.baselineTeam import OffensiveReflexAgent

#from contest.agents.team_name_1.version2.defensive_agent import BeliefBTDefensiveAgent
from contest.keyboard_agents import KeyboardAgent

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [BeliefBTDefensiveAgent2(first_index), OffensiveReflexAgent(second_index) ]