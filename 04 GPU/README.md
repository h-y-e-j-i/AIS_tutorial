# 04 GPU 사용하기
## GPU 인식
- ray에서 GPU을 인식하고 있는지 아래 코드를 통해 확인한다
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
