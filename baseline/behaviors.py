from .behavior_tree import BehaviorNode

class HasFood(BehaviorNode):
    def __init__(self, min_food=1):
        self.min_food = min_food
    
    def execute(self, agent, game_state):
        my_state = game_state.get_agent_state(agent.index)
        return 'SUCCESS' if my_state.num_carrying >= self.min_food else 'FAILURE'

class EnemyNearby(BehaviorNode):
    def __init__(self, distance=5):
        self.distance = distance
    
    def execute(self, agent, game_state):
        my_pos = game_state.get_agent_position(agent.index)
        enemies = [game_state.get_agent_state(i) 
                  for i in agent.get_opponents(game_state)]
        
        for enemy in enemies:
            if enemy.get_position() and not enemy.is_pacman:
                dist = agent.get_maze_distance(my_pos, enemy.get_position())
                if dist <= self.distance:
                    return 'SUCCESS'
        return 'FAILURE'

class AttackFood(BehaviorNode):
    def execute(self, agent, game_state):
        food_list = agent.get_food(game_state).as_list()
        if not food_list:
            return 'FAILURE'
        
        my_pos = game_state.get_agent_position(agent.index)
        closest_food = min(food_list, 
                          key=lambda f: agent.get_maze_distance(my_pos, f))
        
        agent.target = closest_food
        agent.behavior = "ATTACKING"
        return 'SUCCESS'

class FleeToSafety(BehaviorNode):
    # Implementation here
    pass

class ReturnHome(BehaviorNode):
    # Implementation here
    pass