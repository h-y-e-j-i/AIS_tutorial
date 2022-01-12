# 강화학습
## 강화학습 알고리즘
## 학습환경
### Single Agent
``` python
import os, sys, gym, gym.spaces, numpy as np
# 'SUMO_HOME' 환경변수 확인
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

# gym.Env를 상속하여 구현한다
class SumoEnvironment(gym.Env):
    # 주요 API
    #   reset, step, render, close, seed
    #   SUMO 환경에서 강화학습을 구현할 때에는 reset, step 함수만 작성했다.
    #   순서 : init => reset => step => reset => step => ...
    # 설정해야할 attribute
    #   action_space : action의 크기. 
    #   observation_space : observation 크기
    #   reward_range : reward의 범위 (이미 [~∞, ∞]로 정의되어 있다. 그래서 값을 설정할 필요가 없다.)
    
    def __init__(self):
      # 생성자
      self.sim_max_time = 1000
      sumoInitCmd = ['sumo', '-c', '/home/hyejiserver/hyeji/sumo/cross.sumocfg']
      traci.start(sumoInitCmd)
      self.lane_IDs = traci.lane.getIDList()
      self.ts_IDs = traci.trafficlight.getIDList()
      traci.close()

      print(self.lane_IDs)
      print(self.ts_IDs)

      # observation_space, self.action_space는 gym.spaces을 사용하여 최솟값, 최댓값을 정해준다
      # observation_space는 최솟값이 0 행렬, 최댓값이 무한대 행렬이다
      self.observation_space_len = 1 + len(self.lane_IDs) + len(self.lane_IDs) # 행렬의 길이       
      self.observation_space = gym.spaces.Box( low=np.zeros(self.observation_space_len), high=np.array(['inf']*self.observation_space_len))
      # action_space는 0, 1, 2, 3
      self.action_space = gym.spaces.Discrete(4) # gym.spaces.Discrete는 고정된 범위 안에서 음이 아닌 숫자들을 허용한 공간이다. 
    
    def reset(self):
      # 환경 초기화하고 초기 observation을 반환한다      
      # Returns : 
      #   observatin : 초기 observation
      observatin = [0, 0, 0, 0] # 초기 observation
      return observatin

    def step(self, action):
      # 환경을 한 단계씩 진행
      # Args : 
      #   action : agent가 제공하는 action
      # Returns :
      #   observation(object) : 현재 환경의 observation
      #   reward(float) : 이전 action에 대한 reward
      #   done(boolean) : 에피소드가 끝났는지, 계속 진행되야 하는지에 대한 결과
      #   info (dict) : 보조 정보를 포함하고 있다 (디버깅이나 배울 때 유용하다는데... 잘 모르겠다)
      
      traci.simulationStep() # sumo 시뮬레이션 진행
      
      # action
      traci.trafficlight.setPhase('0', action) # 신호등 변경
      
      # state, reward
      state = self.compute_observation(action) # observation 계산
      reward = self.comptue_reward() # reward 계산
      
      # done
      # 시뮬레이션 시간이 정해진 시간을 지나지 않았다면
      if traci.simulation.getTime() < self.sim_max_time:
          done = False # 계속 진행
      else:
      # 지났다면
          done = True # 에피소드 종료 => reset()
          traci.close()

      # info
      info = {}

      return state, reward, done, info
    def render(self, mode='human'):
        # 환경을 랜더링한다
        # SUMO 환경에서 강화학학습 구현할 때에는 사용하지 않았다
    
    def close(self):
        # 프로그램이 종료됐을 때 자동으로 close() 함수에 접근한다
        
    def seed(self, seed=None):
        # 난수 생성기인 시드를 설정한다
        
    
    # observation 계산 
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
    
    # reward 계산 함수
    def comptue_reward(self):
      sum_waiting_time = 0

      for veh_ID in traci.vehicle.getIDList():
          sum_waiting_time += traci.vehicle.getWaitingTime(veh_ID)
      return -sum_waiting_time
```

``` python
import ray
from sumoEnv import SumoEnvironment # 강화학습 환경을 가져온다
from ray.rllib.agents import dqn # 강화학습 알고리즘을 가져온다
from ray.tune.logger import pretty_print

ray.init() # ray 실행

# 강화학습 환경에 필요한 파라미터가 없는 경우에는 env에 환경 클래스 명을 넣어도 된다
trainer = dqn.DQNTrainer(env=SumoEnvironment) 

# 훈련 시작
while(True):
    result = trainer.train()
    print(pretty_print(result)) # 훈련이 됐을 때마다 결과 출력
```
### Multi Agent

```python
from math import inf
from ray.rllib.env import MultiAgentEnv
import gym, gym.spaces, traci, sys, numpy as np

# train할 sumo 환경

NUM_ROUTE_FILE = 50
DELTA_TIME = 3
OBS_LEN = 25
ACT_LEN = 4

class traffic_signal:
    def __init__(self):
        self.current_phase_id = None
        self.next_phase_id = None
        self.yellow_run_time = 0 # 황색불 점등 시간

# MultiAgentEnv을 상속받아 구현한다
class SumoCase03TrainMultiEnvironment(MultiAgentEnv):      
    # 주요 API
    #   reset, step, render,
    #   SUMO 환경에서 강화학습을 구현할 때에는 reset, step 함수만 작성했다.
    #   순서 : init => reset => step => reset => step => ...
    def __init__(self, actions_are_logits, use_gui, sim_max_time, net_file, route_file, algorithm):
        # 생성자    
        self._algorithm = algorithm # 사용한 알고리즘 종류
        self._net_file = net_file # sumo net file path
        self._route_file = route_file # sumo route file path
        self._sim_max_time=sim_max_time # 한 에피소드당 시뮬레이션 진행 최대 시간
        self._sim_episode = 0 # 에피소드 횟수
        # MADDPG emits action logits instead of actual discrete actions
        self._actions_are_logits = actions_are_logits # DDPG를 사용할 때만 True
        # self.sim_max_episoide = sim_max_episoide        
        self._use_gui=use_gui # gui 사용 여부
        if self._use_gui:self._sumoBinary = "sumo-gui"
        else : self._sumoBinary = "sumo"    

        self._traffic_signal = dict() # 신호등

        # self.config_file = config_file
        # self.use_gui = use_gui
        # self.sim_time = sim_time

        # route file 이름 속 번호(1~50)
        # episode가 진행될때마다 1씩 늘어난다.
        self._route_file_num = 1

        # 초기에 sumo를 실행하여 lane의 아이디를 가져온다
        #_sumoInitCmd = ['sumo', '-n', self._net_file, '-r', self._route_file]
        _sumoInitCmd = ['sumo', '-n', self._net_file]
        traci.start(_sumoInitCmd)

        self._ts_IDs = traci.trafficlight.getIDList() # 모든 신호등 ID
        self._lane_IDs = traci.lane.getIDList() # 모든 lane ID
        self._lane_IDs_by_ts = dict() # 신호등별 lane
        self._inlane_IDs_by_ts = dict() # 신호등별 진입 lane
        self._outlane_IDs_by_ts = dict() # 신호들별 진출 lane

        self._phases_info = dict() # 신호등별 phase info


        for ts_ID in self._ts_IDs:
            self._phases_info[ts_ID] = list()      
            self._inlane_IDs_by_ts[ts_ID]=list(traci.trafficlight.getControlledLanes(ts_ID)) # tuple => list 
            self._outlane_IDs_by_ts[ts_ID] = list()

            # 신호등별 logic을 가져온다
            Logic = traci.trafficlight.getAllProgramLogics(ts_ID)
            # 신호등별 programID에 따른 phase를 가져온다                 
            for program in Logic:                            
                for phase in program.getPhases():
                    self._phases_info[ts_ID].append(phase.duration)

            for lanes in traci.trafficlight.getControlledLinks(ts_ID): # 차선1, 차선2, 차선1과 차선2를 잇는 링크
                # 차선1와 차선2가 진입 lane에 없다면 진출 lane임
                if lanes[0][0] not in self._inlane_IDs_by_ts[ts_ID] and lanes[0][0] not in self._outlane_IDs_by_ts[ts_ID]:
                    self._outlane_IDs_by_ts[ts_ID].append(lanes[0][0])
                if lanes[0][1] not in self._inlane_IDs_by_ts[ts_ID] and lanes[0][1] not in self._outlane_IDs_by_ts[ts_ID]:
                    self._outlane_IDs_by_ts[ts_ID].append(lanes[0][1])
                    
            # 신호등별 lane = 신호등별 진입 lane + 진출 lane                       
            self._lane_IDs_by_ts[ts_ID] = self._inlane_IDs_by_ts[ts_ID]+self._outlane_IDs_by_ts[ts_ID]
        traci.close()

        # observation과 action의 크기를 지정

        # 1. aciton + 각 lane의 차량 대수 + 각 lane의 차량들의 총 waiting time
        # self._observation_space_len = 1 + len(self._lane_IDs) + len(self._lane_IDs) 

        # 2. action + agent별 각 lane의 차량 대수 + agent별 각 lane의 차량들의 총 waiting time
        self._observation_space_len = 1 + len(self._lane_IDs_by_ts['gneJ00']) + len(self._lane_IDs_by_ts['gneJ00']) 
        # self._observation_space_len = 1 + len(self._lane_IDs_by_ts['gneJ00'])
        self._observation_space = gym.spaces.Box(low=np.zeros(self._observation_space_len), high=np.array(['inf']*self._observation_space_len))
        self._action_space = gym.spaces.Discrete(len(self._ts_IDs)) 
       
        print(self._phases_info)

    def reset(self):
        # 에피소드가 시작하기 직전 초기화
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        
        """
        # 초기 observation은 0행렬
        state = dict()
        self._reward_sum = dict()

        for ts_ID in self._ts_IDs:
            state[ts_ID] = np.zeros(self._observation_space_len)      
            self._reward_sum[ts_ID] = 0
            self._traffic_signal[ts_ID] = traffic_signal()
        self._reward_sum['sum'] = 0

        # route file의 확장자 8글자 (.rou.xml)를 제거하고 숫자를 추가한 뒤 확장자를 다시 붙인다.
        run_route_file = self._route_file[:-8] + str(self._route_file_num) + '.rou.xml'

        # route file을 50번까지 사용했다면 다시 1부터 다시 시작
        self._route_file_num = (self._route_file_num)%NUM_ROUTE_FILE+1

       # sumo (재)실행
        sumoCmd = [self._sumoBinary, "-n",  self._net_file, '-r', run_route_file, '--quit-on-end', '--random', "--start"]
        traci.start(sumoCmd)     

        self._done = False

        # 에피소드 1회씩 증가
        self._sim_episode += 1
            
        return state

    def step(self, actions):
        # 환경을 한 단계씩 진행
        # Args : 
        #   action : agent가 제공하는 action
        # Returns :
        #   observation(dict) : 현재 환경의 agent id별 observation을 dict 형태로 반환한다. 반환하지 않으면 다음 step에서 반환하지 않은 agent id를 제외하고 action을 받아온다
        #                         예) {'gneJ00':100} 이 반환하면 다음 step에서 gneJ9를 제외한 gneJ00의 action을 받아온다
        #   reward(dict) : 이전 action에 대한 agent id별 reward를 dict 형태로 반환한다
        #   done(dict) : 에피소드가 끝났는지, 계속 진행되야 하는지에 대한 결과. key "__all__"는 환경 종료를 가리킨다
        #   info (dict) :  agent id별 보조 정보를 포함하고 있다 (위에서 말했듯이 잘 모르겠다)
        
        # aciton
        # action : 현재는 3번

        # 1. action은 신호의 phase 번호와 같다
        # for ts_ID in self._ts_IDs:       
        #     traci.trafficlight.setPhase(ts_ID, actions[ts_ID])

        # 2. 청색불만 점등한다. 청색불은 phase 번호가 짝수이므로 action*2 번째 신호를 점등.
        # 신호등 점등 시간이 최소 점등 시간을 넘지 못 할 때
        # for ts_ID in self._ts_IDs:       
        #     traci.trafficlight.setPhase(ts_ID, actions[ts_ID]*2)

        # 3. 청색불일 때만 action O, 황색불일 때는 action X
        for ts_ID in self._ts_IDs:                      
            if ts_ID in actions.keys():    
                # 시뮬레이션이 처음 돌 때                
                if self._traffic_signal[ts_ID].current_phase_id is None:                                
                    self._traffic_signal[ts_ID].current_phase_id = actions[ts_ID]*2 # aciton*2번 청색불이 켜짐
                # 청색불 => 동일한 청색불 : 유지
                elif self._traffic_signal[ts_ID].current_phase_id == actions[ts_ID]*2:
                    pass
                # 청색 불 => 다른 청색불
                elif self._traffic_signal[ts_ID].current_phase_id%2 == 0:                
                    self._traffic_signal[ts_ID].current_phase_id = (self._traffic_signal[ts_ID].current_phase_id+1)%(ACT_LEN*2)  # 다음 황색불로 변경             
                    self._traffic_signal[ts_ID].yellow_run_time = 1 # 황색불 진행 시간 증가
                    self._traffic_signal[ts_ID].next_phase_id = actions[ts_ID]*2 # 다음 phase id = action*2(청색불)
            # #  황색불인경우
            # elif self._traffic_signal[ts_ID].current_phase_id%2 == 1 :                
            else:                
                # 최소시간이 지나지 않았다면 
                if self._traffic_signal[ts_ID].yellow_run_time < self._phases_info[ts_ID][self._traffic_signal[ts_ID].current_phase_id]:
                    self._traffic_signal[ts_ID].yellow_run_time += 1 # 황색불 진행시간만 증가 
                # 최소시간이 지났다면                 
                else:
                    self._traffic_signal[ts_ID].current_phase_id = self._traffic_signal[ts_ID].next_phase_id # 다음 청색불로 변경
                    self._traffic_signal[ts_ID].yellow_run_time = 0 # 황색불 진행 시간 초기화
                    self._traffic_signal[ts_ID].next_phase_id = None # 다음 phase id None       
                    
            traci.trafficlight.setPhase(ts_ID, self._traffic_signal[ts_ID].current_phase_id)

        traci.simulationStep()

                   
        states = self._compute_local_observation(actions) # observation 계산          
        rewards = self._compute_reward() # reward 계산       

        dones = dict()
        infos = dict()
        for ts_ID in self._ts_IDs:     
            if self._traffic_signal[ts_ID].current_phase_id%2 == 0:       
                dones[ts_ID] = False
                infos[ts_ID] = {}            
                self._reward_sum[ts_ID] += rewards[ts_ID]
                self._reward_sum['sum'] += rewards[ts_ID]

        # if traci.simulation.getTime()%100 == 0:
        #     print(states)
        #     print(rewards)

        if traci.simulation.getTime() <=self._sim_max_time: # 현재 시뮬레이션 시간이 시뮬레이션 최대 시간보다 작다면 = 아직 에피소드가 끝나지 않았다면 
           # 계속 진행           
            dones['__all__'] = False
        else: # 그렇지 않다면
            dones['__all__'] = True  # 에피소드 종료
            print(self._reward_sum)
            traci.close() # sumo 종료          

        return states, rewards, dones, infos
        
    
    # reward 계산 메소드

    # 1. -(차량들의 총 대기시간)
    # def _compute_reward(self):
    #     sum_waiting_time = 0 
    #     reward = dict()

    #     for veh_ID in traci.vehicle.getIDList():
    #         # 차량들의 총 대기시간
    #         sum_waiting_time += traci.vehicle.getWaitingTime(veh_ID)

    #     for ts_ID in self._ts_IDs:
    #         reward[ts_ID] = -sum_waiting_time
    #     return reward

    # 2. 지나간 차량대수 - 멈춘 차량대수
    def _compute_reward(self):        
        rewards = dict()        

        for ts_ID in self._ts_IDs:
            halting_vehicle = 0 # 진입 lane에서 멈춘 차량 
            passed_vehicle = 0 # 진출 lane으로 지나간 차량
             # 청색불인 경우만 reward을 구한다
            if self._traffic_signal[ts_ID].current_phase_id%2 == 0 :
                for outlane_ID in self._outlane_IDs_by_ts[ts_ID]:
                    # 진출 lane으로 지나간 차량 = 진출 lane에서 마지막 차량 대수 - 진출 lane에서 정지 차량 대수
                    passed_vehicle += traci.lane.getLastStepVehicleNumber(outlane_ID)
                    passed_vehicle -= traci.lane.getLastStepHaltingNumber(outlane_ID)
                
                for inlane_ID in self._inlane_IDs_by_ts[ts_ID]:
                    halting_vehicle += traci.lane.getLastStepHaltingNumber(inlane_ID)

                rewards[ts_ID] = passed_vehicle-halting_vehicle
        
        return rewards
       
    #  observation 계산
    # observation 계산 메소드
    # # 1. 현재 action + 전체 lane별 차량 대기 시간의 합 + 전체 lane별 정치 차량 대수
    # def _compute_global_observation(self, action):        
    #     state = dict()        
    #     waiting_vehicle_time_state = []
    #     halting_vehicle_number_state = []

    #     for lane_ID in self._lane_IDs:           
    #         waiting_vehicle_time = 0
    #         for veh_ID in traci.lane.getLastStepVehicleIDs(lane_ID):
    #             waiting_vehicle_time += traci.vehicle.getWaitingTime(veh_ID)
    #         waiting_vehicle_time_state.append(waiting_vehicle_time) # c
    #         halting_vehicle_number_state.append(traci.lane.getLastStepVehicleNumber(lane_ID)) # 각 lane별 정지 차량 대수

    #     for ts_ID in self._ts_IDs:
    #         state[ts_ID] = list()
    #         state[ts_ID].append(action[ts_ID])
    #         state[ts_ID] = state[ts_ID]+waiting_vehicle_time_state+halting_vehicle_number_state
    #     # observation = action + 각 lane별 차량 대기 시간의 합 +  각 lane별 정지 차량 대수
        
    #     #state = state+waiting_vehicle_time_state+halting_vehicle_number_state 
    #     return state

    #  observation 계산
    def _compute_local_observation(self, action):        
        state = dict()        

        for ts_ID in self._ts_IDs:         
            # 청색불인 경우만 observation을 구한다
            if self._traffic_signal[ts_ID].current_phase_id%2 == 0:   
                state[ts_ID] = list()
                waiting_vehicle_time_state = []
                halting_vehicle_number_state = []

                # # action
                # state[ts_ID].append(action[ts_ID])

                # current phase id
                state[ts_ID].append(self._traffic_signal[ts_ID].current_phase_id)

                for lane_ID in self._lane_IDs_by_ts[ts_ID]:           
                    waiting_vehicle_time = 0
                    # 각 lane에 있는 차량들의 대기시간의 합
                    for veh_ID in traci.lane.getLastStepVehicleIDs(lane_ID):                    
                        waiting_vehicle_time += traci.vehicle.getWaitingTime(veh_ID)                    
                    waiting_vehicle_time_state.append(waiting_vehicle_time)

                    # 각 lane별 정지 차량 대수
                    halting_vehicle_number_state.append(traci.lane.getLastStepVehicleNumber(lane_ID))

                # observation = action + 각 lane별 차량 대기 시간의 합 +  각 lane별 정지 차량 대수            
                state[ts_ID] = state[ts_ID]+waiting_vehicle_time_state+halting_vehicle_number_state
        return state

```
```python
import ray, numpy as np
import tensorflow
from gym import spaces
from case03TrainMultiEnv_random import SumoCase03TrainMultiEnvironment as SumoCase03TrainMultiEnvironment_random
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


def policy_mapping(id):
    return id

OBS_LEN = 25
ACT_LEN = 4

if __name__ == "__main__":
    ray.init() # ray를 실행한다
    
    # 강화학습 클래스에 파라미터가 필요한 경우 register_env을 사용한다(밑에서 설명)
    register_env("sumoCase03TrainMultiEnv_random", lambda _:SumoCase03TrainMultiEnvironment_random(
                                        net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case03/intersection_random.rou.xml',
                                        use_gui=False,
                                        algorithm="DQN",
                                        sim_max_time=3600,
                                        actions_are_logits=False))
    
    stop_timesteps = 500000
    
    # Multi Agent 환경이므로 Trainer의 설정에서 "multiagent"를 추가하여 설정한다
    trainer = dqn.DQNTrainer(env="sumoCase03TrainMultiEnv_random", config={
        "multiagent": {
            "policies": {  #"policy_graphs" # policy : agent가 어떻게 행동할지 결정한다
                # 편의상 policy 이름을 agent 이름과 일치시켰다.
                # 각 policy에 해당하는 agent의 observation 크기와 action 크기를 정한다
                'gneJ00': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(OBS_LEN), high=np.array(['inf']*OBS_LEN)), spaces.Discrete(ACT_LEN), {}),
                'gneJ9': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(OBS_LEN), high=np.array(['inf']*OBS_LEN)), spaces.Discrete(ACT_LEN), {}), 
            },            
            # agent id을 policy id에 매핑한다
            "policy_mapping_fn": policy_mapping  # Traffic lights are always controlled by this policy
        },
        #"timesteps_per_iteration" : stop_timesteps,
        "lr": 0.0001,
    })
    
    # 훈련
    iter = 100000000000000
    for i in range(iter):        
        result = trainer.train()   
        print(pretty_print(result))
```
## 학습 환경의 파라미터 유무
### 파라미터가 없는 경우
```python
import gym

class SumoEnvironment(gym.Env):
    def __init__(self):
        ...
```
와 같이 파라미터가 없는 경우
```python
from sumoEnv import SumoEnvironment
from ray.rllib.agents import dqn 

# env에 환경 클래스 명만 넣어도 된다
trainer = dqn.DQNTrainer(env=SumoEnvironment) 
```
### 파라미터가 있는 경우
```python
class SumoCase03TrainMultiEnvironment(MultiAgentEnv):      
    def __init__(self, actions_are_logits, use_gui, sim_max_time, net_file, route_file, algorithm):
        ...
```
와 같이 학습 환경에 파라미터가 필요한 경우에는 register env을 사용한다
```python
from ray.tune.registry import register_env
from case03TrainMultiEnv_random import SumoCase03TrainMultiEnvironment as SumoCase03TrainMultiEnvironment_random
from ray.rllib.agents import dqn 

# register_env("환경 이름", lambda_: 환경 클래스 이름( 파라미터 설정)
register_env("sumoCase03TrainMultiEnv_random", lambda _:SumoCase03TrainMultiEnvironment_random(
                                        net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case03/intersection_random.rou.xml',
                                        use_gui=False,
                                        algorithm="DQN",
                                        sim_max_time=3600,
                                        actions_are_logits=False))
                                        
# Sigle Agent와 다르게 env에는 register_env에서 정해준 환경 이름을 넣는다
trainer = dqn.DQNTrainer(env="sumoCase03TrainMultiEnv_random", config={
        "multiagent": {
            "policies": {  #"policy_graphs" 
                'gneJ00': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(OBS_LEN), high=np.array(['inf']*OBS_LEN)), spaces.Discrete(ACT_LEN), {}),
                'gneJ9': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(OBS_LEN), high=np.array(['inf']*OBS_LEN)), spaces.Discrete(ACT_LEN), {}), 
            },            
            "policy_mapping_fn": policy_mapping  # Traffic lights are always controlled by this policy
        },
        #"timesteps_per_iteration" : stop_timesteps,
        "lr": 0.0001,
    })
```
