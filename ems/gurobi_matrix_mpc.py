if __name__ == "__main__":
    from manager import Manager
    from logger_manager import log_scenarios, log_real_power
else:
    from ems.manager import Manager
    from ems.logger_manager import log_scenarios, log_real_power

from scipy.optimize import linprog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp


class GurobiMatrixMPC(Manager):
    def __init__(self, fixed_steps, weight_step="equal", steps_skip=1, file_name=None):
        super().__init__()

        self.fixed_steps = fixed_steps
        self.file_name = file_name
        # Can be equal, cufe or favour_next
        self.weight_step = weight_step
        self.steps_skip = steps_skip
        self.steps_cache = [[] for _ in range(steps_skip)]

    def calculate_powers(self, observation, forec_scenarios, time_step):
        A_ub_val = list()
        A_ub_col = list()
        A_ub_row = list()
        cur_ub_row = 0

        A_eq_val = list()
        A_eq_col = list()
        A_eq_row = list()
        cur_eq_row = 0

        index_cache = time_step % self.steps_skip

        if index_cache != 0:
            cached_action = self.steps_cache[index_cache]
            return np.array(cached_action)
        else:
            self.steps_cache = [[] for _ in range(self.steps_skip)]

        num_scenarios = len(forec_scenarios)
        num_buildings = len(forec_scenarios[0])
        horizon = len(forec_scenarios[0][0])

        batt_capacity = [6.4 for i in range(num_buildings)]
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
                building_steps = [
                    forec_scenarios[i][k][j] for k in range(num_buildings)
                ]
                forec_step_sum[i].append(sum(building_steps))

        soc_init = [observation[i][22] * batt_capacity[i] for i in range(num_buildings)]

        last_step_nobatt = [
            observation[i][20] - observation[i][21] for i in range(num_buildings)
        ]
        last_step_batt = [observation[i][23] for i in range(num_buildings)]
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
            for j in range(num_buildings):
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

        non_fixed_steps = horizon - fixed_steps
        num_obj = 3 * num_scenarios  # Positive
        num_fixed_pos = fixed_steps * num_buildings  # Positive
        num_fixed_neg = fixed_steps * num_buildings  # Negative
        num_mult_pos = non_fixed_steps * num_scenarios * num_buildings  # Positive
        num_mult_neg = non_fixed_steps * num_scenarios * num_buildings  # Negative
        num_carb_pow = horizon * num_scenarios * num_buildings  # Positive
        num_price_pow = horizon * num_scenarios  # Positive
        num_grid_abs_1 = horizon * num_scenarios  # Positive
        num_grid_abs_2 = horizon * num_scenarios  # Positive

        num_var = (
            num_obj
            + num_fixed_pos
            + num_fixed_neg
            + num_mult_pos
            + num_mult_neg
            + num_carb_pow
            + num_price_pow
            + num_grid_abs_1
            + num_grid_abs_2
        )
        fixed_pos_level = num_obj
        fixed_neg_level = fixed_pos_level + num_fixed_pos
        mult_pos_level = fixed_neg_level + num_fixed_neg
        mult_neg_level = mult_pos_level + num_mult_pos
        carb_level = mult_neg_level + num_mult_neg
        price_level = carb_level + num_carb_pow
        grid_abs_1_level = price_level + num_price_pow
        grid_abs_2_level = grid_abs_1_level + num_grid_abs_1

        # Objective function First 3 variables is value used for each cost
        obj_func = np.zeros((num_var,), dtype=float)
        for i in range(num_scenarios):
            obj_func[i * 3] = 1 / base_carb[i] / 3
            obj_func[i * 3 + 1] = 1 / base_price[i] / 3
            obj_func[i * 3 + 2] = 1 / base_grid[i] / 6

        # Set carbon cost to be positive
        # forall s,b,t batt_power[s,b,t]-carb_pow[s,b,t]/carb_cost[t]<=-baseload[s,b,t]
        # or (batt_power[s,b,t]+baseload[s,b,t])*carb_cost[t]<=carb_pow[s,b,t]
        # carb_pos_constr_ub = list()
        carb_pos_equal_ub = list()

        for i in range(num_scenarios):
            for j in range(num_buildings):
                for k in range(horizon):
                    # cur_constr = np.zeros((num_var,), dtype=float)

                    # Select battery power
                    if k < fixed_steps:
                        fixed_incr = j * fixed_steps + k
                        var_ind = fixed_pos_level + fixed_incr
                        # cur_constr[var_ind] = 1
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind)
                        A_ub_val.append(1)

                        var_ind = fixed_neg_level + fixed_incr
                        # cur_constr[var_ind] = 1

                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind)
                        A_ub_val.append(1)
                    else:
                        ind_to_add = k - fixed_steps
                        mult_incr = (
                            i * num_buildings * non_fixed_steps
                            + j * non_fixed_steps
                            + ind_to_add
                        )
                        var_ind = mult_pos_level + mult_incr
                        # cur_constr[var_ind] = 1
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind)
                        A_ub_val.append(1)

                        var_ind = mult_neg_level + mult_incr
                        # cur_constr[var_ind] = 1
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind)
                        A_ub_val.append(1)

                    # Select positive carbon value and divide by the carbon costs
                    var_ind = carb_level + i * num_buildings * horizon + j * horizon + k
                    # cur_constr[var_ind] = -1 / carb_cost[k]
                    A_ub_row.append(cur_ub_row)
                    A_ub_col.append(var_ind)
                    A_ub_val.append(-1 / carb_cost[k])

                    cur_equal = -forec_scenarios[i][j][k]

                    # carb_pos_constr_ub.append(cur_constr)
                    cur_ub_row += 1
                    carb_pos_equal_ub.append(cur_equal)

        # Sum all carbon costs to get the final carbon cost per scenario.
        # forall s tot_carb[s] = sum_[b,t] {carb_pow[s,b,t]}
        # carb_calc_constr_eq = list()
        carb_calc_equal_eq = list()
        for i in range(num_scenarios):
            # cur_constr = np.zeros((num_var,), dtype=float)
            var_ind = i * 3
            # cur_constr[var_ind] = 1
            A_eq_row.append(cur_eq_row)
            A_eq_col.append(var_ind)
            A_eq_val.append(1)

            for j in range(num_buildings):
                for k in range(horizon):
                    var_ind = carb_level + i * num_buildings * horizon + j * horizon + k
                    # cur_constr[var_ind] = -1 / weight_steps[k]
                    A_eq_row.append(cur_eq_row)
                    A_eq_col.append(var_ind)
                    A_eq_val.append(-1 / weight_steps[k])

            cur_equal = 0
            # carb_calc_constr_eq.append(cur_constr)
            cur_eq_row += 1
            carb_calc_equal_eq.append(cur_equal)

        # Set price cost to be positive
        # forall s,t sum_b{batt_power[s,b,t]}-price_pow[s,t]/price_cost[t]<=sum_b{-baseload[s,b,t]}
        # or sum_b{batt_power[s,b,t]+baseload[s,b,t]}*price_cost[t]<=price_pow[s,t]
        # price_pos_constr_ub = list()
        price_pos_equal_ub = list()

        for i in range(num_scenarios):
            for k in range(horizon):
                # cur_constr = np.zeros((num_var,), dtype=float)
                for j in range(num_buildings):
                    # Select the battery power
                    if k < fixed_steps:
                        fixed_incr = j * fixed_steps + k
                        var_ind = fixed_pos_level + fixed_incr
                        # cur_constr[var_ind] = 1
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind)
                        A_ub_val.append(1)
                        var_ind = fixed_neg_level + fixed_incr
                        # cur_constr[var_ind] = 1
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind)
                        A_ub_val.append(1)
                    else:
                        ind_to_add = k - fixed_steps
                        mult_incr = (
                            i * num_buildings * non_fixed_steps
                            + j * non_fixed_steps
                            + ind_to_add
                        )
                        var_ind = mult_pos_level + mult_incr
                        # cur_constr[var_ind] = 1
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind)
                        A_ub_val.append(1)
                        var_ind = mult_neg_level + mult_incr
                        # cur_constr[var_ind] = 1
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind)
                        A_ub_val.append(1)

                # Select positive price value and divide by the price costs
                var_ind = price_level + i * horizon + k
                # cur_constr[var_ind] = -1 / price_cost[k]
                A_ub_row.append(cur_ub_row)
                A_ub_col.append(var_ind)
                A_ub_val.append(-1 / price_cost[k])

                cur_equal = -sum(forec_scenarios[i][b][k] for b in range(num_buildings))

                # price_pos_constr_ub.append(cur_constr)
                cur_ub_row += 1
                price_pos_equal_ub.append(cur_equal)

        # Sum all price costs to get the final price cost per scenario.
        # forall s tot_price[s] = sum_[t]{price_pow[s,t]}
        # price_calc_constr_eq = list()
        price_calc_equal_eq = list()
        for i in range(num_scenarios):
            # cur_constr = np.zeros((num_var,), dtype=float)
            var_ind = i * 3 + 1
            # cur_constr[var_ind] = 1
            A_eq_row.append(cur_eq_row)
            A_eq_col.append(var_ind)
            A_eq_val.append(1)

            for k in range(horizon):
                var_ind = price_level + i * horizon + k
                # cur_constr[var_ind] = -1 / weight_steps[k]
                A_eq_row.append(cur_eq_row)
                A_eq_col.append(var_ind)
                A_eq_val.append(-1 / weight_steps[k])
            cur_equal = 0
            # price_calc_constr_eq.append(cur_constr)
            cur_eq_row += 1
            price_calc_equal_eq.append(cur_equal)

        # Grid cost absolute constraint, general idea:
        # for all s,t sum_b{(forec[s,b,t]+batt_pow[s,b,t])-(forec[s,b,t-1]+batt_pow[s,b,t-1])}=abs1[s,t]-abs2[s,t]
        # Implementation most time step sum_b{-batt_pow[s,b,t]+batt_pow[s,b,t-1]}+abs1[s,t]-abs2[s,t]= forec[s,t]-forec[s,t-1]
        # Implementation first time step sum_b{-batt_pow[s,b,t]}+abs1[s,t]-abs2[s,t]=forec[s,t]-last_step_pow
        # abs_constr_eq = list()
        abs_equal_eq = list()
        for i in range(num_scenarios):
            for k in range(horizon):
                # cur_constr = np.zeros((num_var,), dtype=float)
                if k == 0:
                    equal_constr = forec_step_sum[i][k] - last_step_batt_sum
                else:
                    equal_constr = forec_step_sum[i][k] - forec_step_sum[i][k - 1]

                for j in range(num_buildings):
                    fixed_incr = j * fixed_steps + k

                    ind_to_add = k - fixed_steps
                    mult_incr = (
                        i * non_fixed_steps * num_buildings
                        + j * non_fixed_steps
                        + ind_to_add
                    )
                    if k < fixed_steps:
                        prev_batt_pow_pos_ind = fixed_pos_level + fixed_incr - 1
                        prev_batt_pow_neg_ind = fixed_neg_level + fixed_incr - 1
                        batt_pow_pos_ind = fixed_pos_level + fixed_incr
                        batt_pow_neg_ind = fixed_neg_level + fixed_incr
                    elif k == fixed_steps:
                        prev_batt_pow_pos_ind = fixed_pos_level + fixed_incr - 1
                        prev_batt_pow_neg_ind = fixed_neg_level + fixed_incr - 1

                        batt_pow_pos_ind = mult_pos_level + mult_incr
                        batt_pow_neg_ind = mult_neg_level + mult_incr
                    else:
                        prev_batt_pow_pos_ind = mult_pos_level + mult_incr - 1
                        prev_batt_pow_neg_ind = mult_neg_level + mult_incr - 1

                        batt_pow_pos_ind = mult_pos_level + mult_incr
                        batt_pow_neg_ind = mult_neg_level + mult_incr
                    if k != 0:
                        # cur_constr[prev_batt_pow_pos_ind] = 1
                        A_eq_row.append(cur_eq_row)
                        A_eq_col.append(prev_batt_pow_pos_ind)
                        A_eq_val.append(1)
                        # cur_constr[prev_batt_pow_neg_ind] = 1
                        A_eq_row.append(cur_eq_row)
                        A_eq_col.append(prev_batt_pow_neg_ind)
                        A_eq_val.append(1)
                    # cur_constr[batt_pow_pos_ind] = -1
                    A_eq_row.append(cur_eq_row)
                    A_eq_col.append(batt_pow_pos_ind)
                    A_eq_val.append(-1)

                    # cur_constr[batt_pow_neg_ind] = -1
                    A_eq_row.append(cur_eq_row)
                    A_eq_col.append(batt_pow_neg_ind)
                    A_eq_val.append(-1)

                abs1_ind = grid_abs_1_level + i * horizon + k
                abs2_ind = grid_abs_2_level + i * horizon + k

                # cur_constr[abs1_ind] = 1
                A_eq_row.append(cur_eq_row)
                A_eq_col.append(abs1_ind)
                A_eq_val.append(1)
                # cur_constr[abs2_ind] = -1
                A_eq_row.append(cur_eq_row)
                A_eq_col.append(abs2_ind)
                A_eq_val.append(-1)

                # abs_constr_eq.append(cur_constr)
                cur_eq_row += 1
                abs_equal_eq.append(equal_constr)

        # Sum all grid costs to get the final frid cost per scenario.
        # forall s tot_grid[s] = sum_[t]{abs1[s,t]+abs2[s,t]}
        # grid_calc_constr_eq = list()
        grid_calc_equal_eq = list()
        for i in range(num_scenarios):
            # cur_constr = np.zeros((num_var,), dtype=float)
            var_ind = i * 3 + 2
            # cur_constr[var_ind] = 1
            A_eq_row.append(cur_eq_row)
            A_eq_col.append(var_ind)
            A_eq_val.append(1)
            for k in range(horizon):
                abs1_ind = grid_abs_1_level + i * horizon + k
                abs2_ind = grid_abs_2_level + i * horizon + k
                var_ind = price_level + i * horizon + k
                # cur_constr[abs1_ind] = -1 / weight_steps[k]
                A_eq_row.append(cur_eq_row)
                A_eq_col.append(abs1_ind)
                A_eq_val.append(-1 / weight_steps[k])
                # cur_constr[abs2_ind] = -1 / weight_steps[k]
                A_eq_row.append(cur_eq_row)
                A_eq_col.append(abs2_ind)
                A_eq_val.append(-1 / weight_steps[k])
            cur_equal = 0
            # grid_calc_constr_eq.append(cur_constr)
            cur_eq_row += 1
            grid_calc_equal_eq.append(cur_equal)

        # SOC contraints
        # General idea:
        # for all s,b,t 0<=soc_init[b]+(batt_power[s,b,0]+...+batt_power[s,b,t]<=6.4
        # (batt_power[s,b,0]+...+batt_power[s,b,t]<=6.4-soc_init
        # -(batt_power[s,b,0]+...+batt_power[s,b,t])<=soc_init[b]
        # -soc_init[b]-(batt_power[s,b,0]+...+batt_power[s,b,t])<=0
        # !!! Added efficiency to power after comment
        # soc_low_constr_ub = list()
        soc_low_equal_ub = list()

        # soc_high_constr_ub = list()
        soc_high_equal_ub = list()

        for i in range(num_scenarios):
            for j in range(num_buildings):
                for k in range(horizon):
                    # cur_constr_low = np.zeros((num_var,), dtype=float)
                    # cur_constr_high = np.zeros((num_var,), dtype=float)
                    for l in range(k + 1):
                        if l < fixed_steps:
                            fixed_incr = j * fixed_steps + l
                            var_ind_pos = fixed_pos_level + fixed_incr
                            var_ind_neg = fixed_neg_level + fixed_incr
                        else:
                            ind_to_add = l - fixed_steps
                            mult_incr = (
                                i * non_fixed_steps * num_buildings
                                + j * non_fixed_steps
                                + ind_to_add
                            )

                            var_ind_pos = mult_pos_level + mult_incr
                            var_ind_neg = mult_neg_level + mult_incr

                        # cur_constr_low[var_ind_pos] = -batt_efficiency
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind_pos)
                        A_ub_val.append(-batt_efficiency)
                        # cur_constr_low[var_ind_neg] = -1 / batt_efficiency
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind_neg)
                        A_ub_val.append(-1 / batt_efficiency)

                    equal_constr_low = soc_init[j]
                    # soc_low_constr_ub.append(cur_constr_low)
                    cur_ub_row += 1
                    soc_low_equal_ub.append(equal_constr_low)
        for i in range(num_scenarios):
            for j in range(num_buildings):
                for k in range(horizon):
                    # cur_constr_high = np.zeros((num_var,), dtype=float)
                    for l in range(k + 1):
                        if l < fixed_steps:
                            fixed_incr = j * fixed_steps + l
                            var_ind_pos = fixed_pos_level + fixed_incr
                            var_ind_neg = fixed_neg_level + fixed_incr
                        else:
                            ind_to_add = l - fixed_steps
                            mult_incr = (
                                i * non_fixed_steps * num_buildings
                                + j * non_fixed_steps
                                + ind_to_add
                            )

                            var_ind_pos = mult_pos_level + mult_incr
                            var_ind_neg = mult_neg_level + mult_incr
                        # cur_constr_high[var_ind_pos] = batt_efficiency
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind_pos)
                        A_ub_val.append(batt_efficiency)
                        # cur_constr_high[var_ind_neg] = 1 / batt_efficiency
                        A_ub_row.append(cur_ub_row)
                        A_ub_col.append(var_ind_neg)
                        A_ub_val.append(1 / batt_efficiency)

                    equal_constr_high = batt_capacity[j] - soc_init[j]

                    # soc_high_constr_ub.append(cur_constr_high)
                    cur_ub_row += 1
                    soc_high_equal_ub.append(equal_constr_high)

        c = np.array(obj_func, dtype=float)
        """
        A_ub = np.array(
            carb_pos_constr_ub
            + price_pos_constr_ub
            + soc_low_constr_ub
            + soc_high_constr_ub,
            dtype=float,
        )
        """
        b_ub = np.array(
            carb_pos_equal_ub
            + price_pos_equal_ub
            + soc_low_equal_ub
            + soc_high_equal_ub,
            dtype=float,
        )
        """
        A_eq = np.array(
            carb_calc_constr_eq
            + price_calc_constr_eq
            + abs_constr_eq
            + grid_calc_constr_eq,
            dtype=float,
        )
        """
        b_eq = np.array(
            carb_calc_equal_eq
            + price_calc_equal_eq
            + abs_equal_eq
            + grid_calc_equal_eq,
            dtype=float,
        )

        rest_positive = num_carb_pow + num_price_pow + num_grid_abs_1 + num_grid_abs_2

        fixed_pos_bound = [(0, max_power) for _ in range(num_fixed_pos)]
        fixed_neg_bound = [(-max_power, 0) for _ in range(num_fixed_neg)]

        for i, soc in enumerate(soc_init):
            if soc > 0.8 * batt_capacity[i]:
                new_max_power = 1
                fixed_pos_bound[i * fixed_steps] = (0, new_max_power)
                fixed_neg_bound[i * fixed_steps] = (-new_max_power, 0)

        bounds = (
            [(0, None) for _ in range(num_obj)]
            + fixed_pos_bound
            + fixed_neg_bound
            + [(0, max_power) for _ in range(num_mult_pos)]
            + [(-max_power, 0) for _ in range(num_mult_neg)]
            + [(0, None) for _ in range(rest_positive)]
        )
        """
        bounds = (
            [(0, None) for _ in range(num_obj)]
            + [(0, max_power) for _ in range(num_fixed_pos)]
            +[(-max_power, 0) for _ in range(num_fixed_neg)]
            +[(0, None) for _ in range(num_mult_pos)]
            +[(None, 0) for _ in range(num_mult_neg)]
            + [(0, None) for _ in range(rest_positive)]
        )
        """
        A_ub_sp = [np.array(A_ub_val), np.array(A_ub_row), np.array(A_ub_col)]
        A_eq_sp = [np.array(A_eq_val), np.array(A_eq_row), np.array(A_eq_col)]

        x = scipy_to_gurobi(
            c,
            A_ub=None,
            b_ub=b_ub,
            A_eq=None,
            b_eq=b_eq,
            bounds=bounds,
            A_ub_sp=A_ub_sp,
            A_eq_sp=A_eq_sp,
        )
        # soc_power.append(res)
        # print(res)
        actions = list()
        power_batteries = [[] for _ in range(num_buildings)]
        for i in range(num_buildings):
            for j in range(fixed_steps):
                power_batteries[i].append(
                    (
                        x[fixed_pos_level + i * fixed_steps + j]
                        + x[fixed_neg_level + i * fixed_steps + j]
                    )
                )

            # Cache actions
            for j, _ in enumerate(self.steps_cache):
                cached_action = power_batteries[i][j] / batt_capacity[i]
                self.steps_cache[j].append([cached_action])

            action = power_batteries[i][0] / batt_capacity[i]
            actions.append([action])
        actions = np.array(actions)

        # fun_score = res.fun + 1 / 6
        # Debug single forecast MPC with scores and all
        if False:
            ### Debug
            base_costs = {
                "base_carb": base_carb[0],
                "base_price": base_price[0],
                "base_grid": base_grid[0],
            }
            log_powers_mpc_perfect(
                self.file_name,
                res.x,
                forec_scenarios,
                time_step,
                base_costs,
                fixed_neg_level,
            )
            powers = [val[0] * batt_capapowerscity[i] for i, val in enumerate(actions)]
            ### End debug

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


def make_sparse(A):
    A_flat = A.flatten()

    row_A_ub = np.repeat(np.arange(A.shape[0]), A.shape[1])
    col_A_ub = row_A_ub.reshape(A.shape[1], A.shape[0], order="F").ravel()

    # np.repeat(np.arange(A.shape[1]), A.shape[1]).reshape(
    #    A.shape[1], A.shape[1], order="F"
    # ).ravel()
    # .reshape(3,3,order='F').ravel()

    row_A_ub = np.zeros(A_flat.size)
    col_A_ub = np.zeros(A_flat.size)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            index = i * A.shape[0] + j
            row_A_ub[index] = i
            col_A_ub[index] = j

    A_sparse = sp.csr_matrix((A_flat, (row_A_ub, col_A_ub)), shape=A.shape)
    return A_sparse


def scipy_to_gurobi(c, A_ub, b_ub, A_eq, b_eq, bounds, A_ub_sp, A_eq_sp):
    m = create_model()
    # Create variables
    lb = [i[0] if i[0] is not None else -GRB.INFINITY for i in bounds]
    ub = [i[1] if i[1] is not None else GRB.INFINITY for i in bounds]
    x = m.addMVar(shape=len(bounds), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")

    # Set objective
    obj = c
    m.setObjective(obj @ x, GRB.MINIMIZE)

    bound_len = len(bounds)
    A_ub_sparsed, A_eq_sparsed = sparsed_time(A_ub_sp, A_eq_sp, bound_len)

    # Add constraints
    # m.addConstr(A_ub_sparsed @ x <= b_ub, name="c1")
    # m.addConstr(A_eq_sparsed @ x == b_eq, name="c2")
    add_constraints(A_ub_sparsed, b_ub, A_eq_sparsed, b_eq, x, m)

    # Optimize model
    solve_problem(m)
    return x.X


def create_model():
    # Create a new model
    m = gp.Model("matrix1")
    m.Params.LogToConsole = 0
    return m


def add_constraints(A_ub_sparsed, b_ub, A_eq_sparsed, b_eq, x, m):
    m.addConstr(A_ub_sparsed @ x <= b_ub, name="c1")
    m.addConstr(A_eq_sparsed @ x == b_eq, name="c2")


def solve_problem(m):
    m.optimize()


def sparsed_time(A_ub_sp, A_eq_sp, bound_len):
    # Build (sparse) constraint matrix
    row_eq_len = max(A_eq_sp[1]) + 1
    row_ub_len = max(A_ub_sp[1]) + 1
    col_leng = bound_len

    A_ub_sparsed = sp.csr_matrix(
        (A_ub_sp[0], (A_ub_sp[1], A_ub_sp[2])), shape=(row_ub_len, col_leng)
    )
    # A_ub_to_comp = A_ub_sparsed.todense()
    A_eq_sparsed = sp.csr_matrix(
        (A_eq_sp[0], (A_eq_sp[1], A_eq_sp[2])), shape=(row_eq_len, col_leng)
    )
    return A_ub_sparsed, A_eq_sparsed
    pass


if __name__ == "__main__":
    plot_logs_mpc_perfect("debug_logs/mpc_debug_pow_0.csv")
