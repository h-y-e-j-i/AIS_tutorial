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
    
    # reward 계산
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

trainer = dqn.DQNTrainer(env=SumoEnvironment)

# 훈련 시작
while(True):
    result = trainer.train()
    print(pretty_print(result)) # 훈련이 됐을 때마다 결과 출력
```
