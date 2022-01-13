import ray, os
@ray.remote(num_gpus=4)
def use_gpu():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

ray.init(num_gpus=4)
use_gpu.remote()