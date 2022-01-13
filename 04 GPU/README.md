# 04 GPU 사용하기
## GPU 인식
- ray는 병렬처리를 위해 ray.remote라는 데코레이터를 클래스나 함수에 붙여서 사용한다
- 먼저 ray에서 GPU을 인식하고 있는지 ray_gpu_detect_test.py 통해 확인한다
```python
import ray, os
@ray.remote(num_gpus=4)
def use_gpu():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

ray.init(num_gpus=4)
use_gpu.remote()
```
```
(pid=36943) ray.get_gpu_ids(): [0, 1, 2, 3]
(pid=36943) CUDA_VISIBLE_DEVICES: 0,1,2,3
```

- 기본적으로, ray.remote 데코레이터는 함수에 사용한다.
- 클래스에 ray.remote 데코레이터를 사용하면 액터라고 부른다.
- ray.remote를 사용하는 방법은 다음과 같다
```python
import ray

# 함수
@ray.remote(num_gpus=4)
def def_remote():
    print("def")

#클래스
@ray.remote(num_gpus=4)
class class_remote():
    def print(self):
        print("class")
        
ray.init(num_gpus=4) # gpu를 사용하면서 ray 실행
def_remote.remote() # ray.remote 데코레이터를 사용한 함수 호출

actor = class_remote.remote() # ray.remote 데코레이터를 사용한 클래스 호출
actor.print.remote() # 클래스의 함수 호출
```
- 아직 분산처리에 대한 개념이 잡히지 않아서 gpu를  제대로 써보지는 못했습니다. 그래서 아래 블로그 포스트도 참고하면 좋을 것 같습니다
    - https://titania7777.tistory.com/15
    - https://right1203.tistory.com/19
    - https://zzsza.github.io/mlops/2021/01/03/python-ray/
    - https://newly0513.tistory.com/217
