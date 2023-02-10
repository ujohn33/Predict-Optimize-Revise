def log_usefull(env, filename):
    num_time_steps = env.time_step + 1
    num_batt = len(env.buildings)
    body = ",".join([f"charge_power_{i}" for i in range(num_batt)]) + ","
    body += ",".join([f"final_load_{i}" for i in range(num_batt)]) + ","
    body += ",".join([f"baseload_{i}" for i in range(num_batt)]) + ","
    body += "prices,carbon_intensity,co2_eval,co2_eval_no_batt,price_eval,price_eval_no_batt\n"

    buildings = env.unwrapped._CityLearnEnv__buildings
    electricity_pricing = buildings[0].pricing.electricity_pricing
    carbon_intensity_arr = buildings[0].carbon_intensity.carbon_intensity
    net_electricity_consumption_emission = env.net_electricity_consumption_emission
    net_electricity_consumption_without_storage_emission = (
        env.net_electricity_consumption_without_storage_emission
    )
    net_electricity_consumption_price = env.net_electricity_consumption_price

    net_electricity_consumption_without_storage_price = (
        env.net_electricity_consumption_without_storage_price
    )

    net_elec_cons_list = [
        buildings[j].net_electricity_consumption for j in range(num_batt)
    ]
    net_elec_cons_list_withou_charge = [
        buildings[j].net_electricity_consumption_without_storage
        for j in range(num_batt)
    ]

    for i in range(num_time_steps):

        all_ch_pow = [
            str(net_elec_cons_list[j][i] - net_elec_cons_list_withou_charge[j][i])
            for j in range(num_batt)
        ]
        charge_power = ",".join(all_ch_pow)
        body += f"{charge_power},"

        all_fin_load = [str(net_elec_cons_list[j][i]) for j in range(num_batt)]
        final_load = ",".join(all_fin_load)
        body += f"{final_load},"

        all_bas_load = [
            str(net_elec_cons_list_withou_charge[j][i]) for j in range(num_batt)
        ]
        base_load = ",".join(all_bas_load)
        body += f"{base_load},"

        prices = electricity_pricing[i]
        body += f"{prices},"

        carbon_intensity = carbon_intensity_arr[i]
        body += f"{carbon_intensity},"

        co2_eval = net_electricity_consumption_emission[i]
        body += f"{co2_eval},"

        co2_eval_no_batt = net_electricity_consumption_without_storage_emission[i]
        body += f"{co2_eval_no_batt},"

        price_eval = net_electricity_consumption_price[i]
        if price_eval < 0:
            price_eval = 0.0

        body += f"{price_eval},"

        price_eval_no_batt = net_electricity_consumption_without_storage_price[i]
        if price_eval_no_batt < 0:
            price_eval_no_batt = 0.0

        body += f"{price_eval_no_batt}\n"

    file = open(filename, "w+")
    file.write(body)
    file.close()