class BehaviorNode:
    """Base class for all behavior tree nodes"""
    def execute(self, agent, game_state):
        raise NotImplementedError

class Selector(BehaviorNode):
    """Try children until one succeeds (OR node)"""
    def __init__(self, children):
        self.children = children
    
    def execute(self, agent, game_state):
        for child in self.children:
            result = child.execute(agent, game_state)
            if result != 'FAILURE':
                return result
        return 'FAILURE'

class Sequence(BehaviorNode):
    """Execute children in order (AND node)"""
    def __init__(self, children):
        self.children = children
    
    def execute(self, agent, game_state):
        for child in self.children:
            result = child.execute(agent, game_state)
            if result == 'FAILURE':
                return 'FAILURE'
        return 'SUCCESS'