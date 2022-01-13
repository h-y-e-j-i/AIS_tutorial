# 모델 커스텀 하기
## 모델 구조 확인하기
```
result = trainer.train()    
model = trainer3.get_policy().model 
# policy id가 있다면
# model = trainer3.get_policy(policy id).model 
model.base_model.summary()
model.state_value_head.summary()
model.q_value_head.summary()

```
## 모델 그래프 저장하기
```python
result = trainer.train()    
policy_grpah = trainer.get_policy(policy_id).get_session().graph
with tensorflow.compat.v1.Graph().as_default():
writer = tensorflow.compat.v1.summary.FileWriter()
```
