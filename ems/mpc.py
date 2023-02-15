if __name__ == "__main__":
    from manager import Manager
else:
    from ems.manager import Manager
from scipy.optimize import linprog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MPC(Manager):
    def __init__(self, fixed_steps):
        super().__init__()
        
        self.fixed_steps = fixed_steps
        self.file_name = "debug_logs/mpc_debug"

    def calculate_powers(self, observation, forec_scenarios, time_step):

        num_scenarios = len(forec_scenarios)
        num_buildings = len(forec_scenarios[0])
        horizon = len(forec_scenarios[0][0])

        batt_capacity = [6.4 for i in range(num_buildings)]
        nominal_power = [5.0 for i in range(num_buildings)]

        if self.fixed_steps == 0:
            fixed_steps = horizon
        else:
            fixed_steps = self.fixed_steps
        # Calculate the sum of the forecasted building power at each step
        forec_step_sum = list()
        for i in range(num_scenarios):
            forec_step_sum.append(list())
            for j in range(horizon):
                building_steps = [
                    forec_scenarios[i][k][j] for k in range(num_buildings)
                ]
                forec_step_sum[i].append(sum(building_steps))

        soc_init = [observation[i][22] * batt_capacity[i] for i in range(num_buildings)]

        last_step = [observation[i][23] for i in range(num_buildings)]
        last_step_sum = sum(last_step)

        price_cost = [self.price_df[(time_step+1 + i) % 8760] for i in range(horizon)]

        carb_cost = [self.carb_df[(time_step +1+ i) % 8760] for i in range(horizon)]

        # Base cost calculation
        base_grid = list()
        base_price = list()
        base_carb = list()

        for i in range(num_scenarios):
            forec_last_st = [last_step_sum] + forec_step_sum[i]

            base_carb_cost = 0
            for j in range(num_buildings):
                base_carb_cost += sum(
                    [
                        val * carb_cost[k] if val >= 0 else 0
                        for k, val in enumerate(forec_scenarios[i][j])
                    ]
                )
            base_carb.append(base_carb_cost)

            base_price_cost = sum(
                [
                    val * price_cost[j] if val >= 0 else 0
                    for j, val in enumerate(forec_step_sum[i])
                ]
            )
            base_price.append(base_price_cost)

            base_grid_cost = sum(
                [abs(forec_last_st[j + 1] - forec_last_st[j]) for j in range(horizon)]
            )
            base_grid.append(base_grid_cost)

        non_fixed_steps = horizon - fixed_steps
        num_obj = 3 * num_scenarios  # Positive
        num_fixed = fixed_steps * num_buildings  # Real
        num_mult = non_fixed_steps * num_scenarios * num_buildings  # Real
        num_carb_pow = horizon * num_scenarios * num_buildings  # Positive
        num_price_pow = horizon * num_scenarios  # Positive
        num_grid_abs_1 = horizon * num_scenarios  # Positive
        num_grid_abs_2 = horizon * num_scenarios  # Positive

        num_var = (
            num_obj
            + num_fixed
            + num_mult
            + num_carb_pow
            + num_price_pow
            + num_grid_abs_1
            + num_grid_abs_2
        )
        mult_level = num_obj + num_fixed
        carb_level = mult_level + num_mult
        price_level = carb_level + num_carb_pow
        grid_abs_1_level = price_level + num_price_pow
        grid_abs_2_level = grid_abs_1_level + num_grid_abs_1

        # Objective function First 3 variables is value used for each cost
        obj_func = [0 for _ in range(num_var)]
        for i in range(num_scenarios):
            obj_func[i * 3] = 1 / base_carb[i]
            obj_func[i * 3 + 1] = 1 / base_price[i]
            obj_func[i * 3 + 2] = 1 / base_grid[i]

        # Set carbon cost to be positive
        # forall s,b,t batt_power[s,b,t]-carb_pow[s,b,t]/carb_cost[t]<=-baseload[s,b,t]
        # or (batt_power[s,b,t]+baseload[s,b,t])*carb_cost[t]<=carb_pow[s,b,t]
        carb_pos_constr_ub = list()
        carb_pos_equal_ub = list()

        for i in range(num_scenarios):
            for j in range(num_buildings):
                for k in range(horizon):
                    
                    cur_constr = np.zeros((num_var,), dtype=float)

                    # Select battery power
                    if k < fixed_steps:
                        var_ind = num_obj + j * fixed_steps + k
                        cur_constr[var_ind] = 1
                    else:
                        ind_to_add = k - fixed_steps
                        var_ind = (
                            mult_level
                            + i * num_buildings * non_fixed_steps
                            + j * non_fixed_steps
                            + ind_to_add
                        )

                        cur_constr[var_ind] = 1

                    # Select positive carbon value and divide by the carbon costs
                    var_ind = carb_level + i * num_buildings * horizon + j * horizon + k
                    cur_constr[var_ind] = -1 / carb_cost[k]

                    cur_equal = -forec_scenarios[i][j][k]

                    carb_pos_constr_ub.append(cur_constr)
                    carb_pos_equal_ub.append(cur_equal)

        # Sum all carbon costs to get the final carbon cost per scenario.
        # forall s tot_carb[s] = sum_[b,t] {carb_pow[s,b,t]}
        carb_calc_constr_eq = list()
        carb_calc_equal_eq = list()
        for i in range(num_scenarios):
            cur_constr = np.zeros((num_var,), dtype=float)
            var_ind = i * 3
            cur_constr[var_ind] = 1
            for j in range(num_buildings):
                for k in range(horizon):
                    var_ind = carb_level + i * num_buildings * horizon + j * horizon + k
                    cur_constr[var_ind] = -1
            cur_equal = 0
            carb_calc_constr_eq.append(cur_constr)
            carb_calc_equal_eq.append(cur_equal)

        # Set price cost to be positive
        # forall s,t sum_b{batt_power[s,b,t]}-price_pow[s,t]/price_cost[t]<=sum_b{-baseload[s,b,t]}
        # or sum_b{batt_power[s,b,t]+baseload[s,b,t]}*price_cost[t]<=price_pow[s,t]
        price_pos_constr_ub = list()
        price_pos_equal_ub = list()

        for i in range(num_scenarios):
            for k in range(horizon):
                cur_constr = np.zeros((num_var,), dtype=float)
                for j in range(num_buildings):
                    # Select the battery power
                    if k < fixed_steps:
                        var_ind = num_obj + j * fixed_steps + k
                        cur_constr[var_ind] = 1
                    else:
                        ind_to_add = k - fixed_steps
                        var_ind = (
                            mult_level
                            + i * num_buildings * non_fixed_steps
                            + j * non_fixed_steps
                            + ind_to_add
                        )
                    cur_constr[var_ind] = 1
                # Select positive price value and divide by the price costs
                var_ind = price_level + i * horizon + k
                cur_constr[var_ind] = -1 / price_cost[k]

                cur_equal = -sum(forec_scenarios[i][b][k] for b in range(num_buildings))

                price_pos_constr_ub.append(cur_constr)
                price_pos_equal_ub.append(cur_equal)

        # Sum all price costs to get the final price cost per scenario.
        # forall s tot_price[s] = sum_[t]{price_pow[s,t]}
        price_calc_constr_eq = list()
        price_calc_equal_eq = list()
        for i in range(num_scenarios):
            cur_constr = np.zeros((num_var,), dtype=float)
            var_ind = i * 3 + 1
            cur_constr[var_ind] = 1
            for k in range(horizon):
                var_ind = price_level + i * horizon + k
                cur_constr[var_ind] = -1
            cur_equal = 0
            price_calc_constr_eq.append(cur_constr)
            price_calc_equal_eq.append(cur_equal)

        # Grid cost absolute constraint, general idea:
        # for all s,t sum_b{(forec[s,b,t]+batt_pow[s,b,t])-(forec[s,b,t-1]+batt_pow[s,b,t-1])}=abs1[s,t]-abs2[s,t]
        # Implementation most time step sum_b{-batt_pow[s,b,t]+batt_pow[s,b,t-1]}+abs1[s,t]-abs2[s,t]= forec[s,t]-forec[s,t-1]
        # Implementation first time step sum_b{-batt_pow[s,b,t]}+abs1[s,t]-abs2[s,t]=forec[s,t]-last_step_pow
        abs_constr_eq = list()
        abs_equal_eq = list()
        for i in range(num_scenarios):
            for k in range(horizon):
                cur_constr = np.zeros((num_var,), dtype=float)
                if k == 0:
                    equal_constr = forec_step_sum[i][k] - last_step_sum
                else:
                    equal_constr = forec_step_sum[i][k] - forec_step_sum[i][k - 1]

                for j in range(num_buildings):
                    if k < fixed_steps:
                        prev_batt_pow_ind = num_obj + j * fixed_steps + k - 1
                        batt_pow_ind = num_obj + j * fixed_steps + k
                    elif k == fixed_steps:
                        prev_batt_pow_ind = num_obj + j * fixed_steps + k - 1
                        ind_to_add = k - fixed_steps
                        batt_pow_ind = (
                            mult_level + i * non_fixed_steps * num_buildings + j * non_fixed_steps + ind_to_add
                        )
                    else:
                        ind_to_add = k - fixed_steps
                        prev_batt_pow_ind = (
                            mult_level
                            + i * non_fixed_steps * num_buildings
                            + j * non_fixed_steps
                            + ind_to_add
                            - 1
                        )
                        batt_pow_ind = (
                            mult_level + i * non_fixed_steps * num_buildings + j * non_fixed_steps + ind_to_add
                        )
                    if k != 0:
                        cur_constr[prev_batt_pow_ind] = 1
                    cur_constr[batt_pow_ind] = -1

                abs1_ind = grid_abs_1_level + i * horizon + k
                abs2_ind = grid_abs_2_level + i * horizon + k

                cur_constr[abs1_ind] = 1
                cur_constr[abs2_ind] = -1

                abs_constr_eq.append(cur_constr)
                abs_equal_eq.append(equal_constr)
        
        # Sum all grid costs to get the final frid cost per scenario.
        # forall s tot_grid[s] = sum_[t]{abs1[s,t]+abs2[s,t]}
        grid_calc_constr_eq = list()
        grid_calc_equal_eq = list()
        for i in range(num_scenarios):
            cur_constr = np.zeros((num_var,), dtype=float)
            var_ind = i * 3 + 2
            cur_constr[var_ind] = 1
            for k in range(horizon):
                abs1_ind = grid_abs_1_level + i * horizon + k
                abs2_ind = grid_abs_2_level + i * horizon + k
                var_ind = price_level + i * horizon + k
                cur_constr[abs1_ind] = -1
                cur_constr[abs2_ind] = -1
            cur_equal = 0
            grid_calc_constr_eq.append(cur_constr)
            grid_calc_equal_eq.append(cur_equal)

        # SOC contraints
        # General idea:
        # for all s,b,t 0<=soc_init[b]+(batt_power[s,b,0]+...+batt_power[s,b,t]<=6.4
        # (batt_power[s,b,0]+...+batt_power[s,b,t]<=6.4-soc_init
        # -(batt_power[s,b,0]+...+batt_power[s,b,t])<=soc_init[b]

        # -soc_init[b]-(batt_power[s,b,0]+...+batt_power[s,b,t])<=0
        soc_low_constr_ub = list()
        soc_low_equal_ub = list()

        soc_high_constr_ub = list()
        soc_high_equal_ub = list()

        for i in range(num_scenarios):
            for j in range(num_buildings):
                for k in range(horizon):
                    cur_constr_low = np.zeros((num_var,), dtype=float)
                    cur_constr_high = np.zeros((num_var,), dtype=float)
                    for l in range(k+1):
                        if l < fixed_steps:
                            var_ind = num_obj + j * fixed_steps + l
                        else:
                            ind_to_add = l - fixed_steps
                            var_ind = (
                                mult_level
                                + i * non_fixed_steps * num_buildings
                                + j * non_fixed_steps
                                + ind_to_add
                            )

                        cur_constr_low[var_ind] = -1
                        cur_constr_high[var_ind] = 1

                    equal_constr_low = soc_init[j]
                    equal_constr_high = batt_capacity[j] - soc_init[j]

                    soc_low_constr_ub.append(cur_constr_low)
                    soc_low_equal_ub.append(equal_constr_low)

                    soc_high_constr_ub.append(cur_constr_high)
                    soc_high_equal_ub.append(equal_constr_high)

        c = np.array(obj_func,dtype=float)
        A_ub = np.array(
            carb_pos_constr_ub
            + price_pos_constr_ub
            + soc_low_constr_ub
            + soc_high_constr_ub, 
            dtype=float
        )
        b_ub = np.array(
            carb_pos_equal_ub
            + price_pos_equal_ub
            + soc_low_equal_ub
            + soc_high_equal_ub, 
            dtype=float
        )

        A_eq = np.array(carb_calc_constr_eq + price_calc_constr_eq + abs_constr_eq+grid_calc_constr_eq, dtype=float)
        b_eq = np.array(carb_calc_equal_eq + price_calc_equal_eq + abs_equal_eq+grid_calc_equal_eq, dtype=float)

        rest_positive = num_carb_pow + num_price_pow + num_grid_abs_1 + num_grid_abs_2

        bounds = (
            [(0, None) for _ in range(num_obj)]
            + [(None, None) for _ in range(num_fixed + num_mult)]
            + [(0, None) for _ in range(rest_positive)]
        )

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            options={#"maxiter":10,
                # "disp": True
                },
        )
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        # soc_power.append(res)
        # print(res)
        actions = list()
        for i in range(num_buildings):
            action = res.x[num_obj + i*fixed_steps]/batt_capacity[i]
            actions.append([action])
        
        ### Debug
        #base_costs = {"base_carb":base_carb[0],
        #"base_price":base_price[0],
        #"base_grid":base_grid[0]
        #}
        #log_powers_mpc_perfect(self.file_name, res.x, forec_scenarios, time_step, base_costs)
        #powers = [val[0]*batt_capacity[i] for i, val in enumerate(actions)]
        #print(f"step_{time_step}_{powers}")
        ### End debug
        
        return np.array(actions)

def log_powers_mpc_perfect(file_name, result, forec_scenarios, time_step, base_costs):
    
    if time_step%1000 !=0:
        return
    print("Logged mpc")
    file_pow = f"{file_name}_pow_{time_step}.csv"
    num_scenarios = len(forec_scenarios)
    num_buildings = len(forec_scenarios[0])
    horizon = len(forec_scenarios[0][0])

    file = open(file_pow, "w+")
    header_list = [f"base_{i}" for i in range(num_buildings)]+[f"final_{i}" for i in range(num_buildings)]
    header_list += ["base_sum","final_sum"]
    file_content = ",".join(header_list)+"\n"

    for i in range(horizon):
        base_loads = [forec_scenarios[0][j][i] for j in range(num_buildings)]
        total_load = [base_loads[j]+result[num_scenarios*3+horizon*j+i] for j in range(num_buildings)]
        sum_loads = [sum(base_loads),sum(total_load)]

        list_line = [str(j) for j in base_loads+total_load+sum_loads]
        
        file_content += ",".join(list_line)+"\n"

    file.write(file_content)
    file.close()

    file_kpi = f"{file_name}_kpi_{time_step}.csv"
    file = open(file_kpi, "w+")

    file.write("base_carbon,base_price,base_grid,final_carbon,final_price,final_grid\n")
    base_costs = [val for _,val in base_costs.items()]
    final_costs = [result[i] for i in range(3)]
    all_costs = base_costs+final_costs
    file_line = ",".join([str(i) for i in all_costs])
    file.write(file_line)
    file.close()


def plot_logs_mpc_perfect(file_name):
    pd_df = pd.read_csv(file_name)
    #base_cols = [f"base_{i}" for i in range(5)]
    #final_cols = [f"final_{i}" for i in range(5)]
    #pd_df['base_sum'] = pd_df[base_cols].sum(axis=1)
    #pd_df['final_sum'] = pd_df[final_cols].sum(axis=1)

    pd_df[['base_sum','final_sum']].plot()

    plt.show()


if __name__ == "__main__":
    plot_logs_mpc_perfect("debug_logs/mpc_debug_pow_0.csv")
    