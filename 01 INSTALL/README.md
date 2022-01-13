# 1.  설치

## LINUX
### 서버 컴퓨터
- 이미 설치됨
### 로컬 컴퓨터
- WSL(리눅스용 윈도우 하위 시스템)(https://docs.microsoft.com/ko-kr/windows/wsl/install)
	- 윈도우에서 가상머신 없이 리눅스를 사용할 수 있습니다
	- 설치 방법(https://docs.microsoft.com/ko-kr/windows/wsl/install-manual):
		1. powershell을 관리자 권환으로 연다
		2. Linux용 Windows 하위 시스템 옵션 활성화 : 
		dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
		3. virtual machine 기능 사용 : dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
		4. https://docs.microsoft.com/ko-kr/windows/wsl/install-manual 의 4단계에서 최신 패키지를 다운로드한다
		5. WSL2를 기본 버전으로 설정 : wsl --set-default-version 2
		6. Microsoft Store에서 원하는 버전의 Linux을 다운 받는다
- Docker
- Virtual Machine
- VM Ware

## Anaconda
- https://www.anaconda.com/products/individual 에서 다운로드해서 설치
- 주로 사용하는 명령어
	- 가상환경 생성 : conda create -n [가상환경 이름] python='버전'
	- 가상환경 활성화 : conda activate [가상환경 이름]
	- 가상환경 비활성화 : conda deactivate
	- 가상환경에 깔린 패키지 확인 : conda list
- 다운로드 해야 할 패키지들 : 가상 환경안에서 설치하면 가상환경 안에서만, 밖(base)에서 설치하면 밖에 설치된다
	- tensorflow, CUDA, cuDNN : ray에서 gpu를 인식하려면 먼저 tensorflow가 gpu을 인식하는지를 확인해야한다
		1. 터미널에 nvidia-smi 명령어를 입력하고 엔터를 누른다
		2. 밑 사진에서 CUDA version은 설치된 CUDA version이 아닌 추천 버전이다
			- ![image](https://user-images.githubusercontent.com/58590260/148886509-247cb911-f80a-45be-912e-c50555035e3f.png)
		3. CUDA version 확인 : nvcc --version  (https://d33p.tistory.com/16)
			- 확인이 안되면 CUDA가 설치되지 않았다는 뜻이다
		4. cuDNN verison 확인: cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 (https://d33p.tistory.com/16)
			- 확인이 안되면 cuDNN이 설치되지 않았다는 뜻입니다
		5. https://www.tensorflow.org/install/source_windows?hl=ko 에서 tensorflow, CUDA, cuDNN 호환 버전을 확인한다
		6. tf_detect_gpu.py을 실행하여 기기 목록에 gpu가 뜨는지 확인한다
		```python				
		from tensorflow.python.client import device_lib
		print(device_lib.list_local_devices())
		```
		![image](https://user-images.githubusercontent.com/58590260/149270371-a0e66dcd-5869-408a-9b6c-dbc5c26fce41.png)
		
		8. 추가로, tensorflow에 gpu가 할당되는지 확인하고 싶다면, tf_simple_use_gpu.py 을 실행해본다.
		```python
		import tensorflow as tf
		from datetime import datetime

		shape=(int(10000),int(10000))

		with tf.device("/gpu:1"): # 텐서를 1번 gpu에 할당
		    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
		    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
		    sum_operation = tf.reduce_sum(dot_operation)

		startTime = datetime.now()
		with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
			result = session.run(sum_operation)
			print(result)

		print("\n" * 2)
		print("Time taken:", datetime.now() - startTime)
		print("\n" * 2)
		```		
		- 정상적으로 할당됐다면 GPU:번호 가 출력된다.
			- ![image](https://user-images.githubusercontent.com/58590260/149271476-3056a80c-834e-4e0d-b78e-2576e14a5cbf.png)
		- 코드를 실행하기 전에 터미널 창에 nvidia-smi -l 1 을 입력하여 1초마다 nvidia 상태창을 띄워, 실시간으로 GPU 메모리 할당량을 볼 수 있다.
			- ![image](https://user-images.githubusercontent.com/58590260/149271888-2b6b259f-7d94-4ccc-877d-09842fd96249.png)
	- ray[rlib] : ray 강화학습 라이브러리
	- python
## SUMO
### SUMO 설치하기
- 윈도우 : https://sumo.dlr.de/docs/Downloads.php 에서 다운로드해서 설치한다
- 리눅스
	1) 저장소에 있는 패키지 업데이트 : sudo add-apt-repository ppa:sumo/stable
	2) 설치된 패키지들을 최신 버전으로 업데이트 : sudo apt-get update
	3) sumo 설치 : sudo apt-get install sumo sumo-tools sumo-doc
### 환경변수 설정(https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home)
#### 윈도우
- .msi로 설치하면 자동으로 환경변수가 추가됩니다.
- 수동으로 추가하는 방법
	1) 사용자 환경 변수를 추가합니다
	2) 변수 이름 : SUMO_HOME
	3) 변수 값 : sumo 주소
#### 리눅스
- export 명령어를 사용하여 환경변수에 값을 설정할 수 있다. 그런데 재부팅할 때마다 환경 변수 값이 사라지기 때문에 로그인할 매마다 실행되는 .bashrc 파일에 export 명령어을 기입한다
	1) vi 에디터를 사용 : vi ~/.basrhc
	2) 편집모드 들어가서 수정 : i를 누르고 export SUMO_HOME="SUMO 위치"
	3) 명령모드로 들어가서 저장후 강제 종료 저장 : esc를 누르고 :wq! 입력 
	4) .bashrc 실행 : source ~/.bashrc
## 원격과 관련된 프로그램
- Putty(원격 접속 프로그램), Xming(서버의 GUI을 볼 수 있는 프로그램)
- 혹은 MobaXterm(Xming 필요없음. X Window 내장)

			
	

