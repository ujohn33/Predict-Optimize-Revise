def random_policy(observation, action_space):
    return action_space.sample()

class RandomAgent:
    
    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return random_policy(observation, self.action_space[agent_id])

    def set_pv_capacity_and_num_buildings(self, building_info):
        self.num_buildings = len(building_info)
        self.pv_capacity = [i["solar_power"] for i in building_info]
        #self.init_forecast()

    