# 모델 커스텀 하기
## 모델 구조 확인하기
```python
result = trainer.train()    
model = trainer3.get_policy().model 
# policy id가 있다면
# model = trainer3.get_policy(policy id).model 
model.base_model.summary() # Access the base Keras models
model.state_value_head.summary() # Access the Q value model (specific to DQN)
model.q_value_head.summary() # Access the state value model (specific to DQN)
```
```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
observations (InputLayer)       [(None, 25)]         0                                            
__________________________________________________________________________________________________
fc_1 (Dense)                    (None, 256)          6656        observations[0][0]               
__________________________________________________________________________________________________
fc_out (Dense)                  (None, 256)          65792       fc_1[0][0]                       
__________________________________________________________________________________________________
value_out (Dense)               (None, 1)            257         fc_1[0][0]                       
==================================================================================================
Total params: 72,705
Trainable params: 72,705
Non-trainable params: 0
__________________________________________________________________________________________________
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
model_out (InputLayer)       [(None, 256)]             0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 257       
=================================================================
Total params: 66,049
Trainable params: 66,049
Non-trainable params: 0
_________________________________________________________________
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
model_out (InputLayer)          [(None, 256)]        0                                            
__________________________________________________________________________________________________
hidden_0 (Dense)                (None, 256)          65792       model_out[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 4)            1028        hidden_0[0][0]                   
__________________________________________________________________________________________________
tf_op_layer_gneJ00/ones_like/Sh [(2,)]               0           dense[0][0]                      
__________________________________________________________________________________________________
tf_op_layer_gneJ00/ones_like_1/ [(2,)]               0           dense[0][0]                      
__________________________________________________________________________________________________
tf_op_layer_gneJ00/ones_like (T [(None, 4)]          0           tf_op_layer_gneJ00/ones_like/Shap
__________________________________________________________________________________________________
tf_op_layer_gneJ00/ones_like_1  [(None, 4)]          0           tf_op_layer_gneJ00/ones_like_1/Sh
__________________________________________________________________________________________________
tf_op_layer_gneJ00/ExpandDims ( [(None, 4, 1)]       0           tf_op_layer_gneJ00/ones_like[0][0
__________________________________________________________________________________________________
tf_op_layer_gneJ00/ExpandDims_1 [(None, 4, 1)]       0           tf_op_layer_gneJ00/ones_like_1[0]
==================================================================================================
Total params: 66,820
Trainable params: 66,820
Non-trainable params: 0
__________________________________________________________________________________________________
```
## 모델 커스텀 하기
- https://docs.ray.io/en/latest/rllib-models.html#custom-action-distributions
- Tensorflow, PyTorch 둘 다 자세히 다뤄본 적이 없어 적용만 해보고 그 외에는 해보지 못했다
- 
### Tensorflow
``` python

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

class MyModelClass(TFModelV2):
    def __init__(self, *args, **kwargs):
    super(MyModelClass, self).__init__(*args, **kwargs)
    input_layer = tf.keras.layers.Input(...)
    hidden_layer = tf.keras.layers.Dense(...)(input_layer)
    output_layer = tf.keras.layers.Dense(...)(hidden_layer)
    value_layer = tf.keras.layers.Dense(...)(hidden_layer)
    self.base_model = tf.keras.Model(
        input_layer, [output_layer, value_layer])
        
    def forward(self, input_dict, state, seq_lens):
      model_out, self._value_out = self.base_model(
      input_dict["obs"])
      return model_out, state

ModelCatalog.register_custom_model("my_tf_model", MyModelClass)

ray.init()
trainer = ppo.PPOTrainer(env="CartPole-v0", config={
    "model": {
        "custom_model": "my_tf_model",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {},
    },
})
```
### PyTorch
``` python
import torch.nn as nn

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomTorchModel(TorchModelV2):
    def __init__(self, *args, **kwargs):
      TorchModelV2.__init__(self, *args, **kwargs)
      nn.Module.__init__(self)
      self._hidden_layers = nn.Sequential(...)
      self._logits = ...
      self._value_branch = ...
      
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state
        
    def value_function(self): ...

ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)

ray.init()
trainer = ppo.PPOTrainer(env="CartPole-v0", config={
    "framework": "torch",
    "model": {
        "custom_model": "my_torch_model",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {},
    },
})
```
## 모델 그래프 저장하기
```python
result = trainer.train()    
policy_grpah = trainer.get_policy(policy_id).get_session().graph
with tensorflow.compat.v1.Graph().as_default():
writer = tensorflow.compat.v1.summary.FileWriter()
```
## 모델 그래프 확인하기
- 텐서보드에서 확인할 수 있다
- 그래프의 내용이 복잡한데, 아직 각각의 노드들의 의미와 연산을 이해하지 못했다.
```
tensorboard --logdir=path --host localhost
```
![image](https://user-images.githubusercontent.com/58590260/149295945-2ad2681f-9f66-434c-90df-9b296daed2fc.png)

