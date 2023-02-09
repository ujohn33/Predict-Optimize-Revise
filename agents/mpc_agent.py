from ems.mpc import MPC


class MPCAgent:
    def __init__(self):
        self.action_space = {}
        self.prev_steps = {}
        self.manager = MPC(1)
        self.forecaster = ForecastingClass()
        self.time_step = 0

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation):
        """Get observation return action"""
        self.populate_prev_steps(observation)

        forec_scenarios = self.forecaster.get_forecast(self.prev_steps)

        actions = self.manager.calculate_powers(
            self.prev_steps, forec_scenarios, self.time_step
        )
        self.time_step += 1
        return actions

    def set_pv_capacity_and_num_buildings(self, building_info):
        self.num_buildings = len(building_info)
        self.pv_capacity = [i["solar_power"] for i in building_info]
        # self.init_forecast()

    def populate_prev_steps(self, observation):
        ### Put your code here
        pass
