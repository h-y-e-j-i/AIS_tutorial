# 모델 저장하기 & 불러오기
## 모델 저장하기
## 불러오기
- checkpoint를 불러오는 방법은 다음과 같다
```
rllib rollout \
    ~/ray_results/default/DQN_CartPole-v0_0upjmdgr0/checkpoint_1/checkpoint-1 \
    --run DQN --env CartPole-v0 --steps 10000
```
- checkpoint는 홈 디렉토리 밑 ray_results 폴더에 있다. (~./ray_results)
- 훈련한 모델 이름의 폴더로 이동한 후, 불러오고 싶은 checkpoint 폴더에 들어간다
- pwd 명령어를 이용해 현재 위치를 복사하고 불러올 checkpoint 이름(checkpoint-숫자)를 그 뒤에 붙인 파일 경로를 가지고 있는다
  - 예) /home/sonic/ray_results/MADDPG_sumoMultiEnv_2021-09-10_16-59-34n7gahahx/checkpoint_000094/checkpoint-94
- 만약, ray에 내장된 환경이라면 환경 이름을 적으면 되고, 사용자가 만든 환경이라면  /home/sonic/anaconda3/envs/tf_rl/lib/python3.7/site-packages/ray/rllib/rollout.py 파일 안에 from ~ import와 register_env를 사용하여 똑같이 사용자 환경을 만들어 준다.
- 그런데 서로 다른 폴더에 있는 코드다보니 from~import 사용이 조금 까다로워서 그냥  /home/sonic/anaconda3/envs/tf_rl/lib/python3.7/site-packages/ray/rllib 폴더 밑에 학습 환경 코드를 복사하고 register_env를 사용하여 사용자 환경을 등록했다
  - ![image](https://user-images.githubusercontent.com/58590260/149289941-55a5090c-a90f-448f-addf-888bd61128f4.png)
  ```python
  #!/usr/bin/env python

  import argparse
  import collections
  import copy
  import gym
  from gym import wrappers as gym_wrappers
  import json
  import os
  from pathlib import Path
  import shelve

  import ray
  import ray.cloudpickle as cloudpickle
  from ray.rllib.agents.registry import get_trainer_class
  from ray.rllib.env import MultiAgentEnv
  from ray.rllib.env.base_env import _DUMMY_AGENT_ID
  from ray.rllib.env.env_context import EnvContext
  from ray.rllib.evaluation.worker_set import T, WorkerSet
  from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
  from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
  from ray.tune.utils import merge_dicts
  from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
  from ray.tune.registry import register_env
  from .IcheonTestMultiEnv_random_getcontrolledlane import SumoIcheonTestMultiEnvironment as Icheon_random_gcl
  from .IcheonTestMultiEnv_restricted_getcontrolledlane import SumoIcheonTrainMultiEnvironment as Icheon_restricted_gcl
  from .case03TestMultiEnv_random_getcontrolledlane import SumoCase03TrainMultiEnvironment as case03_random_gcl
  from .case03TestMultiEnv_restricted_getcontrolledlane import SumoCase03TrainMultiEnvironment as case03_restricted_gcl
  EXAMPLE_USAGE = """
  Example usage via RLlib CLI:
      rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
      --env CartPole-v0 --steps 1000000 --out rollouts.pkl

  Example usage via executable:
      ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
      --env CartPole-v0 --steps 1000000 --out rollouts.pkl

  Example usage w/o checkpoint (for testing purposes):
      ./rollout.py --run PPO --env CartPole-v0 --episodes 500
  """

  # Note: if you use any custom models or envs, register them here first, e.g.:
  #
  # from ray.rllib.examples.env.parametric_actions_cartpole import \
  #     ParametricActionsCartPole
  # from ray.rllib.examples.model.parametric_actions_model import \
  #     ParametricActionsModel
  # ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
  # register_env("pa_cartpole", lambda _: ParametricActionsCartPole(10))

  register_env("Icheon_random_gcl", lambda _:Icheon_random_gcl(
                                          net_file='/home/sonic/Desktop/nets/case06/intersection_pid0.net.xml',
                                          route_file='/home/sonic/Desktop/nets/case06/intersection_pid2.rou.xml',
                                          use_gui=True,
                                          algorithm="DQN",
                                          actions_are_logits=False,
                                          sim_max_time=3600))

  register_env("Icheon_restricted_gcl", lambda _:Icheon_restricted_gcl(
                                          net_file='/home/sonic/Desktop/nets/case06/intersection_pid1.net.xml',
                                          route_file='/home/sonic/Desktop/nets/case06/intersection_pid1.rou.xml',
                                          use_gui=True,
                                          algorithm="DQN",
                                          actions_are_logits=False,
                                          sim_max_time=3600))

  register_env("case03_random_gcl", lambda _:case03_random_gcl(
                                          net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                          route_file='/home/sonic/Desktop/nets/case03/intersection.rou.xml',
                                          use_gui=True,
                                          algorithm="DQN",
                                          actions_are_logits=False,
                                          sim_max_time=3600))


  register_env("case03_restricted_gcl", lambda _:case03_restricted_gcl(
                                          net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                          route_file='/home/sonic/Desktop/nets/case03/intersection.rou.xml',
                                          use_gui=True,
                                          algorithm="DQN",
                                          actions_are_logits=False,
                                          sim_max_time=3600))


  ...
  ```
  - 테스트 환경에는 기존 환경 조건을 건들지 않는 선에서 코드를 추가해도 된다. (결과 출력하기 등등)
  - 그리고 터미널 창에 훈련했던 알고리즘까지 명령어에 넣으면 된다
   ```
   rllib rollout  /home/sonic/ray_results/DQN_sumoIcheonTrainMultiEnv_random_2021-10-07_16-57-19b27d0vuk/checkpoint_000601/checkpoint-601 --env "Icheon_random_gcl" --run DQN
   ```
## 텐서보드
- 터미널 창에 텐서보드 명령어를 입력한다
```
tensorboard --logdir=~/ray_results --host localhost
```
혹은
```
tensorboard --logdir=checkpoint_path --host localhost
```
- putty나 원격 데스크톱 연결을 통해 접속해 터미널에 firfox라고 입력하여 파이어폭스를 켜 접속한다. 
![image](https://user-images.githubusercontent.com/58590260/149296655-8c05bc7b-244e-4bdf-8c62-8b9df3fa5207.png)
