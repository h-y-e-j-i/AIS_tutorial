import ray

@ray.remote(num_gpus=4)
def def_remote():
    print("def")

@ray.remote(num_gpus=4)
class class_remote():
    def print(self):
        print("class")
        
ray.init(num_gpus=4)
def_remote.remote()

actor = class_remote.remote()
actor.print.remote()