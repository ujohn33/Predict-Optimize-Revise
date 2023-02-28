if __name__ == "__main__":
    from manager import Manager
else:
    from ems.manager import Manager

class LoggerManager(Manager):
    def __init__(self, param=""):
        
        super().__init__()
        self.quantiles_done = {"time_step":-1, "building":-1}
        self.scen_filename = f"debug_logs/scenarios_{param}.csv"
        self.quant_filename = f"debug_logs/quant_next_step_{param}.csv"
        self.real_val_filename = f"debug_logs/real_power_{param}.csv"
    
    def calculate_powers(self, observation, forec_scenarios, time_step):
        
        num_scenarios = len(forec_scenarios)
        num_buildings = len(forec_scenarios[0])
        horizon = len(forec_scenarios[0][0])

        if time_step == 0 :
            scen_file = open(self.scen_filename,"w+")
            
            scen_start = ["time_step","scenario","building",]
            scen_tail = [f"+{i+1}h" for i in range(horizon)]

            scen_head = ",".join(scen_start+scen_tail)+"\n"
            scen_file.write(scen_head)
            scen_file.close()

            real_val_file = open(self.real_val_filename,"w+")
            real_header = ",".join(["time_step"]+[f"building_{i}" for i in range(num_buildings)])
            real_header+= "\n"
            real_val_file.write(real_header)
            real_val_file.close()
        
        scen_file = open(self.scen_filename,"a+")
        for i in range(num_scenarios):
            for j in range(num_buildings):
                line_start = f"{time_step},{i},{j},"
                line_tail = ",".join([str(val) for val in forec_scenarios[i][j]])
                line = line_start+line_tail+"\n"
                scen_file.write(line)
        scen_file.close()

        # Log last step without use of battery
        last_step = [observation[i][20]-observation[i][21] for i in range(num_buildings)]
        real_val_file = open(self.real_val_filename,"a+")
        
        line_start = f"{time_step-1},"
        line_tail = ",".join([str(val) for val in last_step])
        line = line_start+line_tail+"\n"
        real_val_file.write(line)

        real_val_file.close()
        

        return [[0] for _ in range(num_buildings)]

    def log_quantiles(self, quantiles, quantiles_val, time_step, build_num):
        if time_step > self.quantiles_done["time_step"]:
            self.quantiles_done["time_step"] = time_step
            self.quantiles_done["building"] = -1

        if build_num <= self.quantiles_done["building"]:
            return
        else:
            self.quantiles_done["building"] = build_num
        
        num_quant = len(quantiles)
        if time_step == 0 and build_num == 0:
            quant_file = open(self.quant_filename,"w+")
            
            quant_start = ["time_step","building",]
            quant_tail = [str(i) for i in quantiles]

            quant_head = ",".join(quant_start+quant_tail)+"\n"
            quant_file.write(quant_head)
            quant_file.close()

        quant_file = open(self.quant_filename,"a+")
        quant_start = f"{time_step},{build_num},"
        quant_tail = ",".join([str(i) for i in quantiles_val])
        line = quant_start+quant_tail+"\n"
        quant_file.write(line)

        