from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ems.logger_manager import log_real_power, log_scenarios
from ems.manager import Manager
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints

# import logging


class PyoMPC(Manager):
    def __init__(self, fixed_steps, weight_step="equal", steps_skip=1, file_name=None):
        super().__init__()

        self.fixed_steps = fixed_steps
        self.file_name = file_name
        # Can be equal, cufe or favour_next
        self.weight_step = weight_step
        self.steps_skip = steps_skip
        self.steps_cache = [[] for _ in range(steps_skip)]

        self.model = pyo.AbstractModel()
        self.model_initialized = False

        self.opt = pyo.SolverFactory(
            "gurobi",
            options={
                # "OutputFlag": 1,
                "LogToConsole": 1,
            },
        )

    def build_sets(self, horizon, num_batts, num_scen, fixed_steps):
        model = self.model
        time = np.array([i for i in range(horizon)])
        batt_id = np.array([i for i in range(num_batts)])
        scen_id = np.array([i for i in range(num_scen)])
        fixed_step = np.array([i for i in range(fixed_steps)])
        mult_step = np.array([i for i in range(fixed_steps, horizon)])

        model.time = pyo.Set(initialize=time)
        model.batt_id = pyo.Set(initialize=batt_id)
        model.scen_id = pyo.Set(initialize=scen_id)
        model.fixed_step = pyo.Set(initialize=fixed_step)
        model.mult_step = pyo.Set(initialize=mult_step)

    def build_params(self):
        model = self.model
        model.soc_init = pyo.Param(model.batt_id)

        model.last_step_batt_sum = pyo.Param()

        model.price_cost = pyo.Param(model.time)

        model.carb_cost = pyo.Param(model.time)

        model.base_grid = pyo.Param(model.scen_id)

        model.base_carb = pyo.Param(model.scen_id)

        model.base_price = pyo.Param(model.scen_id)

        model.forec_scenarios = pyo.Param(model.scen_id, model.batt_id, model.time)

    def build_vars(self):
        model = self.model

        model.price_obj = pyo.Var(model.scen_id, domain=pyo.PositiveReals)
        model.carb_obj = pyo.Var(model.scen_id, domain=pyo.PositiveReals)
        model.grid_obj = pyo.Var(model.scen_id, domain=pyo.PositiveReals)

        model.fixed_pos = pyo.Var(
            model.batt_id, model.fixed_step, domain=pyo.PositiveReals, bounds=(0, 5)
        )
        model.fixed_neg = pyo.Var(
            model.batt_id, model.fixed_step, domain=pyo.NegativeReals, bounds=(-5, 0)
        )

        model.mult_pos = pyo.Var(
            model.scen_id,
            model.batt_id,
            model.mult_step,
            domain=pyo.PositiveReals,
            bounds=(0, 5),
        )
        model.mult_neg = pyo.Var(
            model.scen_id,
            model.batt_id,
            model.mult_step,
            domain=pyo.NegativeReals,
            bounds=(-5, 0),
        )

        model.carb_ind_cost = pyo.Var(
            model.scen_id, model.batt_id, model.time, domain=pyo.PositiveReals
        )
        model.price_ind_cost = pyo.Var(
            model.scen_id, model.time, domain=pyo.PositiveReals
        )

        model.total_power = pyo.Var(
            model.scen_id, model.batt_id, model.time, domain=pyo.Reals
        )

        model.grid_abs_1 = pyo.Var(model.scen_id, model.time, domain=pyo.PositiveReals)
        model.grid_abs_2 = pyo.Var(model.scen_id, model.time, domain=pyo.PositiveReals)

        model.soc_fixed = pyo.Var(
            model.batt_id,
            model.fixed_step,
            domain=pyo.PositiveReals,
            bounds=(0, 6.4),
        )

        model.soc_mult = pyo.Var(
            model.scen_id,
            model.batt_id,
            model.mult_step,
            domain=pyo.PositiveReals,
            bounds=(0, 6.4),
        )

        model.obj_cost = pyo.Var(domain=pyo.Reals)

    def build_constr(self, batt_eff):
        self.set_total_power_constr()

        self.carbon_pos_constr()
        self.sum_carb_cost_constr()

        self.price_pos_constr()
        self.sum_price_cost_constr()

        self.absolute_grid_diff_constr()
        self.sum_grid_cost_constr()

        self.soc_constr(batt_eff)

        self.obj_cost_constr()

    def build_obj(self):
        model = self.model

        def obj_cost_rule(model):
            return model.obj_cost

        model.obj = pyo.Objective(rule=obj_cost_rule, sense=pyo.minimize)

    def get_instance_data(
        self,
        soc_init,
        last_step_batt_sum,
        price_cost,
        carb_cost,
        base_carb,
        base_price,
        base_grid,
        forec_scenarios,
    ):
        init_data_dict = {None: dict()}

        init_data_dict[None]["soc_init"] = array_to_dict(soc_init)
        init_data_dict[None]["last_step_batt_sum"] = {None: last_step_batt_sum}
        init_data_dict[None]["price_cost"] = array_to_dict(price_cost)
        init_data_dict[None]["carb_cost"] = array_to_dict(carb_cost)
        init_data_dict[None]["base_grid"] = array_to_dict(base_grid)
        init_data_dict[None]["base_carb"] = array_to_dict(base_carb)
        init_data_dict[None]["base_price"] = array_to_dict(base_price)
        init_data_dict[None]["forec_scenarios"] = three_d_array_to_dict(forec_scenarios)

        return init_data_dict

    def set_total_power_constr(self):
        # Set variable with the summed power of consumptiona nd battery
        model = self.model
        scen_id = model.scen_id
        batt_id = model.batt_id
        fixed_step = model.fixed_step
        mult_step = model.mult_step

        def rule_total_power_fixed(model, s, b, t):
            batt_power = model.fixed_pos[b, t] + model.fixed_neg[b, t]
            return model.total_power[s, b, t] == (
                batt_power + model.forec_scenarios[s, b, t]
            )

        model.total_power_fixed = pyo.Constraint(
            scen_id,
            batt_id,
            fixed_step,
            rule=rule_total_power_fixed,
        )

        def rule_total_power_mult(model, s, b, t):
            batt_power = model.mult_pos[s, b, t] + model.mult_neg[s, b, t]
            return model.total_power[s, b, t] == (
                batt_power + model.forec_scenarios[s, b, t]
            )

        model.total_power_mult = pyo.Constraint(
            scen_id,
            batt_id,
            mult_step,
            rule=rule_total_power_mult,
        )

    def carbon_pos_constr(self):
        # Set carbon cost to be positive
        # forall s,b,t batt_power[s,b,t]-carb_pow[s,b,t]/carb_cost[t]<=-baseload[s,b,t]
        # or (batt_power[s,b,t]+baseload[s,b,t])*carb_cost[t]<=carb_ind_cost[s,b,t]
        model = self.model

        scen_id = model.scen_id
        batt_id = model.batt_id
        time = model.time

        def rule_carb_cost(model, s, b, t):
            return (
                model.carb_ind_cost[s, b, t]
                >= model.total_power[s, b, t] * model.carb_cost[t]
            )

        model.carb_cost_fixed = pyo.Constraint(
            scen_id,
            batt_id,
            time,
            rule=rule_carb_cost,
        )

    def sum_carb_cost_constr(self):
        # Sum all carbon costs to get the final carbon cost per scenario.
        # forall s carbon_obj[s] = sum_[b,t] {carb_ind_cost[s,b,t]}
        model = self.model

        scen_id = model.scen_id

        def rule_sum_carb_cost(model, s):
            tot_carb_cost = sum(
                model.carb_ind_cost[s, b, t] for b in model.batt_id for t in model.time
            )
            return model.carb_obj[s] == tot_carb_cost / model.base_carb[s]

        model.sum_carb_cost = pyo.Constraint(
            scen_id,
            rule=rule_sum_carb_cost,
        )

    def price_pos_constr(self):
        # Set price cost to be positive
        # forall s,t sum_b{batt_power[s,b,t]}-price_ind_cost[s,t]/price_cost[t]<=sum_b{-baseload[s,b,t]}
        # or sum_b{batt_power[s,b,t]+baseload[s,b,t]}*price_cost[t]<=price_ind_cost[s,t]
        model = self.model

        scen_id = model.scen_id
        time = model.time

        def rule_price_cost_fixed(model, s, t):
            return (
                model.price_ind_cost[s, t]
                >= sum(model.total_power[s, b, t] for b in model.batt_id)
                * model.price_cost[t]
            )

        model.price_cost_fixed = pyo.Constraint(
            scen_id,
            time,
            rule=rule_price_cost_fixed,
        )

    def sum_price_cost_constr(self):
        # Sum all price costs to get the final price cost per scenario.
        # forall s tot_price[s] = sum_[t]{price_ind_cost[s,t]}
        model = self.model

        scen_id = model.scen_id

        def rule_sum_price_cost(model, s):
            tot_price_cost = sum(model.price_ind_cost[s, t] for t in model.time)
            return model.price_obj[s] == tot_price_cost / model.base_price[s]

        model.sum_price_cost = pyo.Constraint(
            scen_id,
            rule=rule_sum_price_cost,
        )

    def absolute_grid_diff_constr(self):
        # Get the absolute difference between the previous and current power of all timesteps
        # for all s,t sum_b{(forec[s,b,t]+batt_pow[s,b,t])-(forec[s,b,t-1]+batt_pow[s,b,t-1])}=abs1[s,t]-abs2[s,t]
        # Implementation most time step sum_b{-batt_pow[s,b,t]+batt_pow[s,b,t-1]}+abs1[s,t]-abs2[s,t]= forec[s,t]-forec[s,t-1]
        # Implementation first time step sum_b{-batt_pow[s,b,t]}+abs1[s,t]-abs2[s,t]=forec[s,t]-last_step_pow
        model = self.model

        scen_id = model.scen_id
        time = model.time

        def rule_abs_diff(model, s, t):
            if t == 0:
                prev_power = model.last_step_batt_sum
                weight_grid = 1
            else:
                weight_grid = 1
                prev_power = sum(model.total_power[s, b, t - 1] for b in model.batt_id)
            return prev_power - sum(
                model.total_power[s, b, t] for b in model.batt_id
            ) == weight_grid * (model.grid_abs_1[s, t] - model.grid_abs_2[s, t])

        model.abs_diff = pyo.Constraint(
            scen_id,
            time,
            rule=rule_abs_diff,
        )

    def sum_grid_cost_constr(self):
        # Sum up grid cost for each scenario
        # Sum all grid costs to get the final frid cost per scenario.
        # forall s tot_grid[s] = sum_[t]{abs1[s,t]+abs2[s,t]}
        model = self.model

        scen_id = model.scen_id

        def rule_sum_grid_cost(model, s):
            tot_grid_cost = sum(
                model.grid_abs_1[s, t] + model.grid_abs_2[s, t] for t in model.time
            )
            return model.grid_obj[s] == tot_grid_cost / model.base_grid[s]

        model.sum_grid_cost = pyo.Constraint(
            scen_id,
            rule=rule_sum_grid_cost,
        )

    def soc_constr(self, batt_eff):
        model = self.model
        scen_id = model.scen_id
        batt_id = model.batt_id
        fixed_step = model.fixed_step
        mult_step = model.mult_step

        def rule_soc_fixed(model, b, t):
            if t == 0:
                soc_prev = model.soc_init[b]
            else:
                soc_prev = model.soc_fixed[b, t - 1]

            return (
                model.soc_fixed[b, t]
                == soc_prev
                + model.fixed_pos[b, t] * batt_eff
                + model.fixed_neg[b, t] / batt_eff
            )

        model.set_soc_fixed = pyo.Constraint(
            batt_id,
            fixed_step,
            rule=rule_soc_fixed,
        )

        def rule_soc_mult(model, s, b, t):
            if t == model.mult_step[0]:
                soc_prev = model.soc_fixed[b, t]
            else:
                soc_prev = model.soc_mult[s, b, t - 1]

            return (
                model.soc_mult[s, b, t]
                == soc_prev
                + model.mult_pos[s, b, t] * batt_eff
                + model.mult_neg[s, b, t] / batt_eff
            )

        model.set_soc_mult = pyo.Constraint(
            scen_id,
            batt_id,
            mult_step,
            rule=rule_soc_mult,
        )

    def obj_cost_constr(self):
        model = self.model

        def obj_cost_rule(model):
            return model.obj_cost == sum(
                model.carb_obj[s] + model.price_obj[s] + model.grid_obj[s] / 2
                for s in model.scen_id
            )

        model.set_obj_cost = pyo.Constraint(
            rule=obj_cost_rule,
        )

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
            self.build_params()

            self.build_vars()

            self.build_constr(batt_efficiency)
            self.build_obj()
            self.model_initialized = True

        instance_data = self.get_instance_data(
            soc_init,
            last_step_batt_sum,
            price_cost,
            carb_cost,
            base_carb,
            base_price,
            base_grid,
            forec_scenarios,
        )
        model_inst = self.model.create_instance(instance_data)

        # res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        # soc_power.append(res)

        self.opt.solve(model_inst)
        # log_infeasible_constraints(self.model)
        # logging.basicConfig(
        #    filename="example.log", encoding="utf-8", level=logging.INFO
        # )
        actions = list()
        power_batteries = [[] for _ in range(num_batts)]
        for b in range(num_batts):
            for t in range(fixed_steps):
                batt_pos = pyo.value(model_inst.fixed_pos[b, t])
                batt_neg = pyo.value(model_inst.fixed_neg[b, t])
                power_batteries[b].append(batt_pos + batt_neg)

            # Cache actions
            for j, _ in enumerate(self.steps_cache):
                cached_action = power_batteries[b][j] / batt_capacity[b]
                self.steps_cache[j].append([cached_action])

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
