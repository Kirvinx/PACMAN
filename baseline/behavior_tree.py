from contest.game import Directions


class BehaviorNode:
    """Base class for all behavior tree nodes."""
    def execute(self, agent, game_state):
        raise NotImplementedError


class Selector(BehaviorNode):
    """Try children until one succeeds (OR node)."""
    def __init__(self, children):
        self.children = children

    def execute(self, agent, game_state):
        for child in self.children:
            result = child.execute(agent, game_state)

            # If a child returns an action (like 'North'), propagate it up
            if result in Directions.__dict__.values():
                return result

            # If a child succeeds logically, stop here
            if result == 'SUCCESS':
                return 'SUCCESS'

        # If all fail, this node fails
        return 'FAILURE'


class Sequence(BehaviorNode):
    """Execute children in order (AND node)."""
    def __init__(self, children):
        self.children = children

    def execute(self, agent, game_state):
        for child in self.children:
            result = child.execute(agent, game_state)

            # If a child returns an action, bubble it up (end sequence here)
            if result in Directions.__dict__.values():
                return result

            if result == 'FAILURE':
                return 'FAILURE'

        # All children succeeded logically
        return 'SUCCESS'


class Condition(BehaviorNode):
    """
    A condition node checks a function.
    The function should return True/False.
    """
    def __init__(self, condition_func):
        self.condition_func = condition_func

    def execute(self, agent, game_state):
        return 'SUCCESS' if self.condition_func(agent, game_state) else 'FAILURE'


class Action(BehaviorNode):
    """
    An action node performs an action function.
    The function should return:
      - a direction (like Directions.NORTH), OR
      - 'SUCCESS' / 'FAILURE'
    """
    def __init__(self, action_func):
        self.action_func = action_func

    def execute(self, agent, game_state):
        result = self.action_func(agent, game_state)

        # Accept either logical statuses or real movement actions
        if result in Directions.__dict__.values():
            return result
        elif result in ['SUCCESS', 'FAILURE']:
            return result
        else:
            # Defensive default — no action, but not an error
            return 'FAILURE'
