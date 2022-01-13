# 모델 커스텀 하기
## 모델 그래프
```python
result = trainer.train()    
policy_grpah = trainer.get_policy(policy_id).get_session().graph
with tensorflow.compat.v1.Graph().as_default():
writer = tensorflow.compat.v1.summary.FileWriter()
```
