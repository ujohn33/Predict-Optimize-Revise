from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ems.logger_manager import log_real_power, log_scenarios
from ems.manager import Manager
from pyomo.util.infeasible import log_infeasible_constraints
import gurobipy as gp
from gurobipy import GRB


# import logging
class ModelAttributes(object):
    pass


class GurobiMPC(Manager):
    def __init__(self, fixed_steps, weight_step="equal", steps_skip=1, file_name=None):
        super().__init__()

        self.fixed_steps = fixed_steps
        self.file_name = file_name
        # Can be equal, cufe or favour_next
        self.weight_step = weight_step
        self.steps_skip = steps_skip
        self.steps_cache = [[] for _ in range(steps_skip)]

        self.model = gp.Model("citylearn+mpc")
        self.model.Params.LogToConsole = 0
        self.model_att = ModelAttributes()

        self.model_initialized = False

    def build_sets(self, horizon, num_batts, num_scen, fixed_steps):
        model_att = self.model_att
        model_att.time = range(horizon)
        model_att.batt_id = range(num_batts)
        model_att.scen_id = range(num_scen)
        model_att.fixed_step = range(fixed_steps)
        model_att.mult_step = range(fixed_steps, horizon)

    def build_vars(self):
        model = self.model
        model_att = self.model_att

        model_att.price_obj = model.addVars(
            model_att.scen_id, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS
        )
        model_att.carb_obj = model.addVars(
            model_att.scen_id, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS
        )
        model_att.grid_obj = model.addVars(
            model_att.scen_id, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS
        )

        model_att.fixed_pos = model.addVars(
            model_att.batt_id,
            model_att.fixed_step,
            lb=0.0,
            ub=5.0,
            vtype=GRB.CONTINUOUS,
        )
        model_att.fixed_neg = model.addVars(
            model_att.batt_id,
            model_att.fixed_step,
            lb=-5.0,
            ub=0.0,
            vtype=GRB.CONTINUOUS,
        )

        model_att.mult_pos = model.addVars(
            model_att.scen_id,
            model_att.batt_id,
            model_att.mult_step,
            lb=0.0,
            ub=5.0,
            vtype=GRB.CONTINUOUS,
        )
        model_att.mult_neg = model.addVars(
            model_att.scen_id,
            model_att.batt_id,
            model_att.mult_step,
            lb=-5.0,
            ub=0.0,
            vtype=GRB.CONTINUOUS,
        )

        model_att.carb_ind_cost = model.addVars(
            model_att.scen_id,
            model_att.batt_id,
            model_att.time,
            lb=0.0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
        )
        model_att.price_ind_cost = model.addVars(
            model_att.scen_id,
            model_att.time,
            lb=0.0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
        )

        model_att.total_power = model.addVars(
            model_att.scen_id,
            model_att.batt_id,
            model_att.time,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
        )

        model_att.grid_abs_1 = model.addVars(
            model_att.scen_id,
            model_att.time,
            lb=0.0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
        )
        model_att.grid_abs_2 = model.addVars(
            model_att.scen_id,
            model_att.time,
            lb=0.0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
        )

        model_att.soc_fixed = model.addVars(
            model_att.batt_id,
            model_att.fixed_step,
            lb=0.0,
            ub=6.4,
            vtype=GRB.CONTINUOUS,
        )

        model_att.soc_mult = model.addVars(
            model_att.scen_id,
            model_att.batt_id,
            model_att.mult_step,
            lb=0.0,
            ub=6.4,
            vtype=GRB.CONTINUOUS,
        )

        model_att.obj_cost = model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS
        )

    def build_constr(
        self,
        batt_eff,
        forec_scenarios,
        carb_cost,
        base_carb,
        price_cost,
        base_price,
        last_step_batt_sum,
        base_grid,
        soc_init,
    ):
        self.set_total_power_constr(forec_scenarios)

        self.carbon_pos_constr(carb_cost)
        self.sum_carb_cost_constr(base_carb)

        self.price_pos_constr(price_cost)
        self.sum_price_cost_constr(base_price)

        self.absolute_grid_diff_constr(last_step_batt_sum)
        self.sum_grid_cost_constr(base_grid)

        self.soc_constr(batt_eff, soc_init)

        self.obj_cost_constr()

    def build_obj(self):
        model = self.model
        model_att = self.model_att
        model.setObjective(model_att.obj_cost, GRB.MINIMIZE)

    def set_total_power_constr(self, forec_scenarios):
        # Set variable with the summed power of consumptiona nd battery
        model = self.model
        model_att = self.model_att
        scen_id = model_att.scen_id
        batt_id = model_att.batt_id
        fixed_step = model_att.fixed_step
        mult_step = model_att.mult_step

        # Total power fixed
        for s in scen_id:
            for b in batt_id:
                for t in fixed_step:
                    batt_power = model_att.fixed_pos[b, t] + model_att.fixed_neg[b, t]
                    model.addConstr(
                        model_att.total_power[s, b, t]
                        == (batt_power + forec_scenarios[s][b][t])
                    )
        """
        model.addConstrs(
            model_att.total_power[s, b, t]
            == model_att.fixed_pos[b, t]
            + model_att.fixed_neg[b, t]
            + forec_scenarios[s][b][t]
            for s in scen_id
            for b in batt_id
            for t in fixed_step
        )
        model.addConstrs(
            model_att.total_power[s, b, t]
            == model_att.mult_pos[s, b, t]
            + model_att.mult_neg[s, b, t]
            + forec_scenarios[s][b][t]
            for s in scen_id
            for b in batt_id
            for t in mult_step
        )
        """
        # Total power mult
        for s in scen_id:
            for b in batt_id:
                for t in mult_step:
                    batt_power = (
                        model_att.mult_pos[s, b, t] + model_att.mult_neg[s, b, t]
                    )
                    model.addConstr(
                        model_att.total_power[s, b, t]
                        == (batt_power + forec_scenarios[s, b, t])
                    )

    def carbon_pos_constr(self, carb_cost):
        # Set carbon cost to be positive
        # forall s,b,t batt_power[s,b,t]-carb_pow[s,b,t]/carb_cost[t]<=-baseload[s,b,t]
        # or (batt_power[s,b,t]+baseload[s,b,t])*carb_cost[t]<=carb_ind_cost[s,b,t]
        model = self.model
        model_att = self.model_att

        scen_id = model_att.scen_id
        batt_id = model_att.batt_id
        time = model_att.time

        # Carb cost
        for s in scen_id:
            for b in batt_id:
                for t in time:
                    model.addConstr(
                        model_att.carb_ind_cost[s, b, t]
                        >= model_att.total_power[s, b, t] * carb_cost[t]
                    )

    def sum_carb_cost_constr(self, base_carb):
        # Sum all carbon costs to get the final carbon cost per scenario.
        # forall s carbon_obj[s] = sum_[b,t] {carb_ind_cost[s,b,t]}
        model = self.model
        model_att = self.model_att

        scen_id = model_att.scen_id

        # Sum_carb_cost
        for s in scen_id:
            tot_carb_cost = sum(
                model_att.carb_ind_cost[s, b, t]
                for b in model_att.batt_id
                for t in model_att.time
            )
            model.addConstr(model_att.carb_obj[s] == tot_carb_cost / base_carb[s])

    def price_pos_constr(self, price_cost):
        # Set price cost to be positive
        # forall s,t sum_b{batt_power[s,b,t]}-price_ind_cost[s,t]/price_cost[t]<=sum_b{-baseload[s,b,t]}
        # or sum_b{batt_power[s,b,t]+baseload[s,b,t]}*price_cost[t]<=price_ind_cost[s,t]
        model = self.model
        model_att = self.model_att

        scen_id = model_att.scen_id
        time = model_att.time

        # Price cost fixed
        for s in scen_id:
            for t in time:
                model.addConstr(
                    model_att.price_ind_cost[s, t]
                    >= sum(model_att.total_power[s, b, t] for b in model_att.batt_id)
                    * price_cost[t]
                )

    def sum_price_cost_constr(self, base_price):
        # Sum all price costs to get the final price cost per scenario.
        # forall s tot_price[s] = sum_[t]{price_ind_cost[s,t]}
        model = self.model
        model_att = self.model_att

        scen_id = model_att.scen_id

        for s in scen_id:
            tot_price_cost = sum(model_att.price_ind_cost[s, t] for t in model_att.time)
            model.addConstr(model_att.price_obj[s] == tot_price_cost / base_price[s])

    def absolute_grid_diff_constr(self, last_step_batt_sum):
        # Get the absolute difference between the previous and current power of all timesteps
        # for all s,t sum_b{(forec[s,b,t]+batt_pow[s,b,t])-(forec[s,b,t-1]+batt_pow[s,b,t-1])}=abs1[s,t]-abs2[s,t]
        # Implementation most time step sum_b{-batt_pow[s,b,t]+batt_pow[s,b,t-1]}+abs1[s,t]-abs2[s,t]= forec[s,t]-forec[s,t-1]
        # Implementation first time step sum_b{-batt_pow[s,b,t]}+abs1[s,t]-abs2[s,t]=forec[s,t]-last_step_pow
        model = self.model
        model_att = self.model_att

        scen_id = model_att.scen_id
        time = model_att.time

        # Abs diff
        for s in scen_id:
            for t in time:
                if t == 0:
                    prev_power = last_step_batt_sum
                    weight_grid = 1
                else:
                    weight_grid = 1
                    prev_power = sum(
                        model_att.total_power[s, b, t - 1] for b in model_att.batt_id
                    )
                model.addConstr(
                    prev_power
                    - sum(model_att.total_power[s, b, t] for b in model_att.batt_id)
                    == weight_grid
                    * (model_att.grid_abs_1[s, t] - model_att.grid_abs_2[s, t])
                )

    def sum_grid_cost_constr(self, base_grid):
        # Sum up grid cost for each scenario
        # Sum all grid costs to get the final frid cost per scenario.
        # forall s tot_grid[s] = sum_[t]{abs1[s,t]+abs2[s,t]}
        model = self.model
        model_att = self.model_att

        scen_id = model_att.scen_id
        # Sum grid cost

        for s in scen_id:
            tot_grid_cost = sum(
                model_att.grid_abs_1[s, t] + model_att.grid_abs_2[s, t]
                for t in model_att.time
            )
            model.addConstr(model_att.grid_obj[s] == tot_grid_cost / base_grid[s])

    def soc_constr(self, batt_eff, soc_init):
        model = self.model
        model_att = self.model_att

        scen_id = model_att.scen_id
        batt_id = model_att.batt_id
        fixed_step = model_att.fixed_step
        mult_step = model_att.mult_step

        # Soc fixed
        for b in batt_id:
            for t in fixed_step:
                if t == 0:
                    soc_prev = soc_init[b]
                else:
                    soc_prev = model_att.soc_fixed[b, t - 1]
                model.addConstr(
                    model_att.soc_fixed[b, t]
                    == soc_prev
                    + model_att.fixed_pos[b, t] * batt_eff
                    + model_att.fixed_neg[b, t] / batt_eff
                )

        # Soc mult
        for s in scen_id:
            for b in batt_id:
                for t in mult_step:
                    if t == model_att.mult_step[0]:
                        soc_prev = model_att.soc_fixed[b, t]
                    else:
                        soc_prev = model_att.soc_mult[s, b, t - 1]
                    model.addConstr(
                        model_att.soc_mult[s, b, t]
                        == soc_prev
                        + model_att.mult_pos[s, b, t] * batt_eff
                        + model_att.mult_neg[s, b, t] / batt_eff
                    )

    def obj_cost_constr(self):
        model = self.model
        model_att = self.model_att

        model.addConstr(
            model_att.obj_cost
            == sum(
                model_att.carb_obj[s]
                + model_att.price_obj[s]
                 + model_att.grid_obj[s] / 2
                for s in model_att.scen_id
            )
        )

    def model_remove_constr(self):
        for c in self.model.getConstrs():
            self.model.remove(c)

    def calculate_powers(self, observation, forec_scenarios, time_step):
        index_cache = time_step % self.steps_skip

        if index_cache != 0:
            cached_action = self.steps_cache[index_cache]
            return np.array(cached_action)
        else:
            self.steps_cache = [[] for _ in range(self.steps_skip)]

        num_scenarios = len(forec_scenarios)
        num_batts = len(forec_scenarios[0])
        horizon = len(forec_scenarios[0][0])

        batt_capacity = [6.4 for i in range(num_batts)]
        batt_efficiency = 0.91104
        max_power = 5

        if self.fixed_steps == 0:
            fixed_steps = horizon
        else:
            fixed_steps = self.fixed_steps
        # Calculate the sum of the forecasted building power at each step

        if self.weight_step == "equal":
            weight_steps = [1 for _ in range(horizon)]
        elif self.weight_step == "cufe":
            weight_steps = [(46 - i) / 46 for i in range(horizon)]
            weight_steps = [i / sum(weight_steps) * horizon for i in weight_steps]
        elif self.weight_step == "favour_next":
            weight_steps = [1 for _ in range(horizon)]
            weight_steps[0] = 4
            weight_steps = [i / sum(weight_steps) * horizon for i in weight_steps]

        forec_step_sum = list()
        for i in range(num_scenarios):
            forec_step_sum.append(list())
            for j in range(horizon):
                building_steps = [forec_scenarios[i][k][j] for k in range(num_batts)]
                forec_step_sum[i].append(sum(building_steps))

        soc_init = [observation[i][22] * batt_capacity[i] for i in range(num_batts)]

        last_step_nobatt = [
            observation[i][20] - observation[i][21] for i in range(num_batts)
        ]
        last_step_batt = [observation[i][23] for i in range(num_batts)]
        last_step_nobatt_sum = sum(last_step_nobatt)
        last_step_batt_sum = sum(last_step_batt)

        price_cost = [self.price_df[(time_step + 1 + i) % 8760] for i in range(horizon)]

        carb_cost = [self.carb_df[(time_step + 1 + i) % 8760] for i in range(horizon)]

        # Base cost calculation
        base_grid = list()
        base_price = list()
        base_carb = list()

        for i in range(num_scenarios):
            forec_last_st = [last_step_batt_sum] + forec_step_sum[i]

            base_carb_cost = 0
            for j in range(num_batts):
                base_carb_cost += sum(
                    [
                        val * carb_cost[k] if val >= 0 else 0
                        for k, val in enumerate(forec_scenarios[i][j])
                    ]
                )
            if base_carb_cost == 0:
                base_carb_cost = 1
            base_carb.append(base_carb_cost)

            base_price_cost = sum(
                [
                    val * price_cost[j] if val >= 0 else 0
                    for j, val in enumerate(forec_step_sum[i])
                ]
            )
            if base_price_cost == 0:
                base_price_cost = 1
            base_price.append(base_price_cost)

            base_grid_cost = sum(
                [abs(forec_last_st[j + 1] - forec_last_st[j]) for j in range(horizon)]
            )
            base_grid.append(base_grid_cost)

        if not self.model_initialized:
            self.build_sets(horizon, num_batts, num_scenarios, fixed_steps)

            self.build_vars()

            self.build_constr(
                batt_efficiency,
                forec_scenarios,
                carb_cost,
                base_carb,
                price_cost,
                base_price,
                last_step_batt_sum,
                base_grid,
                soc_init,
            )
            self.build_obj()
            self.model_initialized = True

        else:
            # self.opt.reset()
            self.model_remove_constr()
            self.build_constr(
                batt_efficiency,
                forec_scenarios,
                carb_cost,
                base_carb,
                price_cost,
                base_price,
                last_step_batt_sum,
                base_grid,
                soc_init,
            )
            # self.model.update()
            # self.opt_add_constraints()
            """
            param_block = self.build_persistant_params(
                soc_init,
                last_step_batt_sum,
                price_cost,
                carb_cost,
                base_carb,
                base_price,
                base_grid,
                forec_scenarios,
            )
            """
            # self.opt.remove_block(self.model.params)
            # del self.model.params

            # self.model.add_component("params", param_block)
            # self.opt.add_block(self.model.params)

        # res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        # soc_power.append(res)

        self.model.optimize()

        # self.opt.solve(model_inst)
        # log_infeasible_constraints(self.model)
        # logging.basicConfig(
        #    filename="example.log", encoding="utf-8", level=logging.INFO
        # )
        actions = list()
        power_batteries = [[] for _ in range(num_batts)]

        for b in range(num_batts):
            for t in range(fixed_steps):
                batt_pos = self.model_att.fixed_pos[b, t].X
                batt_neg = self.model_att.fixed_neg[b, t].X
                power_batteries[b].append(batt_pos + batt_neg)

            # Cache actions
            for j, _ in enumerate(self.steps_cache):
                cached_action = power_batteries[b][j] / batt_capacity[b]
                self.steps_cache[j].append([cached_action])

            # take an average of xw
            action = power_batteries[b][0] / batt_capacity[b]
            actions.append([action])
        actions = np.array(actions)

        if self.file_name is not None:
            log_real_power(
                time_step, self.file_name.replace("/", "/real_power_"), observation
            )
            log_scenarios(
                time_step, self.file_name.replace("/", "/scen_"), forec_scenarios
            )
            log_fixed_powers(
                time_step, self.file_name.replace("/", "/pow_"), power_batteries
            )
        return actions


def log_powers_mpc_perfect(
    file_name, result, forec_scenarios, time_step, base_costs, fixed_neg_level
):
    file_pow = f"{file_name}_pow_{time_step}.csv"
    num_scenarios = len(forec_scenarios)
    num_buildings = len(forec_scenarios[0])
    horizon = len(forec_scenarios[0][0])

    file = open(file_pow, "w+")
    header_list = [f"base_{i}" for i in range(num_buildings)] + [
        f"final_{i}" for i in range(num_buildings)
    ]
    header_list += ["base_sum", "final_sum"]
    file_content = ",".join(header_list) + "\n"

    for i in range(horizon):
        base_loads = [forec_scenarios[0][j][i] for j in range(num_buildings)]

        total_load = [
            base_loads[j]
            + result[num_scenarios * 3 + horizon * j + i]
            + result[fixed_neg_level + horizon * j + i]
            for j in range(num_buildings)
        ]
        sum_loads = [sum(base_loads), sum(total_load)]

        list_line = [str(j) for j in base_loads + total_load + sum_loads]

        file_content += ",".join(list_line) + "\n"

    file.write(file_content)
    file.close()

    file_kpi = f"{file_name}_kpi_{time_step}.csv"
    file = open(file_kpi, "w+")

    file.write("base_carbon,base_price,base_grid,final_carbon,final_price,final_grid\n")
    base_costs = [val for _, val in base_costs.items()]
    final_costs = [result[i] for i in range(3)]
    all_costs = base_costs + final_costs
    file_line = ",".join([str(i) for i in all_costs])
    file.write(file_line)
    file.close()


def plot_logs_mpc_perfect(file_name):
    pd_df = pd.read_csv(file_name)
    # base_cols = [f"base_{i}" for i in range(5)]
    # final_cols = [f"final_{i}" for i in range(5)]
    # pd_df['base_sum'] = pd_df[base_cols].sum(axis=1)
    # pd_df['final_sum'] = pd_df[final_cols].sum(axis=1)

    pd_df[["base_sum", "final_sum"]].plot()

    plt.show()


def log_fixed_powers(time_step, file_name, power_batteries):
    num_buildings = len(power_batteries)
    fixed_steps = len(power_batteries[0])

    if time_step == 0:
        pow_file = open(file_name, "w+")

        pow_start = [
            "time_step",
            "building",
        ]
        pow_tail = [f"+{i}h" for i in range(fixed_steps)]

        pow_head = ",".join(pow_start + pow_tail) + "\n"
        pow_file.write(pow_head)
        pow_file.close()

    pow_file = open(file_name, "a+")

    for i in range(num_buildings):
        line_start = f"{time_step},{i},"
        line_tail = ",".join([str(val) for val in power_batteries[i]])
        line = line_start + line_tail + "\n"
        pow_file.write(line)

    pow_file.close()


def array_to_dict(array):
    dict_ret = dict()

    for i, elem in enumerate(array):
        dict_ret[i] = elem
    return dict_ret


def two_d_array_to_dict(array):
    dict_ret = dict()
    for i, one_d_array in enumerate(array):
        for j, elem in enumerate(one_d_array):
            dict_ret[(i, j)] = elem

    return dict_ret


def three_d_array_to_dict(array):
    dict_ret = dict()
    for i, two_d_array in enumerate(array):
        for j, one_d_array in enumerate(two_d_array):
            for k, elem in enumerate(one_d_array):
                dict_ret[(i, j, k)] = elem

    return dict_ret


if __name__ == "__main__":
    plot_logs_mpc_perfect("debug_logs/mpc_debug_pow_0.csv")
