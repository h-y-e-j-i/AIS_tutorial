import os, sys, gym, gym.spaces, numpy as np
from gym.core import RewardWrapper
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

class SumoEnvironment(gym.Env):
    def __init__(self):
        self.sim_max_time = 1000

        # 초기에 sumo를 실행하여 lane의 아이디를 가져온다
        sumoInitCmd = ['sumo', '-c', '/home/hyejiserver/hyeji/sumo/cross.sumocfg']
        traci.start(sumoInitCmd)
        self.lane_IDs = traci.lane.getIDList()
        self.ts_IDs = traci.trafficlight.getIDList()
        traci.close()

        print(self.lane_IDs)
        print(self.ts_IDs)

        # observation과 action의 크기를 지정
        # spaces.Box(low=np.zeros(10), high=np.array(['inf']*10)), spaces.Discrete(2), {}) # 1+1+8
        self.observation_space_len = 1 + len(self.lane_IDs) + len(self.lane_IDs)
        self.observation_space = gym.spaces.Box( low=np.zeros(self.observation_space_len), high=np.array(['inf']*self.observation_space_len))
        self.action_space = gym.spaces.Discrete(4)       


    def reset(self):
        state = np.zeros(self.observation_space_len)
        #state = np.array([0, 0, 0, 0])

        if traci.isLoaded():
            traci.close()

        sumoCmd = ['sumo-gui', '-c', '/home/hyejiserver/hyeji/sumo/cross.sumocfg', '--quit-on-end', '--start']
        traci.start(sumoCmd)

        return state

    def step(self, action):
        
        traci.simulationStep()
        traci.trafficlight.setPhase('0', action)  

        state = self.compute_observation(action)
        reward = self.ccomptue_reward()

        if traci.simulation.getTime()%100 == 0:
            print(reward)
            print(state)

        if traci.simulation.getTime() < self.sim_max_time:
            done = False
        else:
            done = True        
            traci.close()

        info = {}

        return state, reward, done, info

    def compute_observation(self, action):
        state = []
        waiting_vehicle_time = 0
        waiting_vehicle_time_state = []
        halting_vehicle_number_state = []

        state.append(action)

        for lane_ID in self.lane_IDs:      
            for veh_ID in traci.lane.getLastStepVehicleIDs(lane_ID):
                waiting_vehicle_time += traci.vehicle.getWaitingTime(veh_ID)
            waiting_vehicle_time_state.append(waiting_vehicle_time)
            halting_vehicle_number_state.append(traci.lane.getLastStepVehicleNumber(lane_ID))

        state = state+waiting_vehicle_time_state+halting_vehicle_number_state
        return np.array(state)

    def ccomptue_reward(self):
        sum_waiting_time = 0

        for veh_ID in traci.vehicle.getIDList():
            sum_waiting_time += traci.vehicle.getWaitingTime(veh_ID)
        return -sum_waiting_time


    

    