import gym, random
import pandas as pd
from gym import spaces
from ray.rllib.agents import dqn

GAME_END = 20 # 가위보위보를 20번하면 에피소드 종료

class RockScissorsPaper(gym.Env):
    # observation : 컴퓨터가 낸 손 모양
    # action : agent가 낸 손 모양
    # reward : agent가 현재 게임에서 획득한 보상
    def __init__(self, env_config):
        self.computer = None
        self.me = None

        # [내가 낸 손 모양, 컴퓨터가 낸 손모양]
        self.victory = [[1, 2], [2, 0], [0, 1]]
        self.fail = [[2, 1], [0, 2], [1, 0]]

        self.game_count = 0  
        self.episode_count = 0

        self.action_space = spaces.Discrete(3) # 0은 주먹, # 1은 가위, 2는 보자기
        self.observation_space = spaces.Discrete(3) # 0은 주먹, # 1은 가위, 2는 보자기

    def reset(self):
        self.computer = random.randrange(0, 3) # 컴퓨터가 초기에 낼 손 모양
        self.observation = self.computer

        self.episode_count += 1
        self.game_count = 0

        return self.observation
    
    def step(self, action):
        self.me = action 
        reward = None
        info = {}

        # reward 계산
        if self.me == self.computer : reward = 1 # 무승부
        elif [self.me, self.computer] in self.victory : reward = 2 # 우승
        elif [self.me, self.computer] in self.fail : reward = 0 # 패배

        observation = random.randrange(0, 3) # 다음 step에 컴퓨터가 낼 손 모양
        
        self.game_count += 1 # 게임 횟수 증가

        # 게임 횟수가 20번 이상이라면 에피소드 종료
        if self.game_count >= GAME_END : 
            done = True
        else : 
            done = False

        #print(self.results)        

        return observation, reward, done, info

if __name__ == "__main__" :
    trainer = dqn.DQNTrainer(env=RockScissorsPaper)
    while True:
        trainer.train()
