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
      ...
      
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

# MultiAgentEnv을 상속받아 구현한다
class SumoCase03TrainMultiEnvironment(MultiAgentEnv):      
    # 주요 API
    #   reset, step, render,
    #   SUMO 환경에서 강화학습을 구현할 때에는 reset, step 함수만 작성했다.
    #   순서 : init => reset => step => reset => step => ...
    def __init__(self, actions_are_logits, use_gui, sim_max_time, net_file, route_file, algorithm):
        # 생성자    
        ...

    def reset(self):
        # 에피소드가 시작하기 직전 초기화
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        
        """
        # 초기 observation은 0행렬
        state = dict()
        self._reward_sum = dict()
            
        ...
            
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
        
        ...      

        return states, rewards, dones, infos


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

## 구현해보면 좋을 간단한 환경
### 가위바위보
- observation : 컴퓨터가 낸 손 모양
- action : agent가 낸 손 모양
- reward : agent가 현재 게임에서 획득한 보상 
### SUMO(3차선)
- observation : 대기 차량 대수
- action : 신호등이 점등할 불  
- reward : 통과한 차량 
