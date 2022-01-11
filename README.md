# AIS_document

	1. LINUX 설치 : 어디에 설치할까?
		a. 서버 컴퓨터 : 이미 설치됨
		b. WSL(리눅스용 윈도우 하위 시스템)(https://docs.microsoft.com/ko-kr/windows/wsl/install) :  윈도우에서 가상머신 없이 리눅스를 사용할 수 있습니다
			i. 설치 방법(https://docs.microsoft.com/ko-kr/windows/wsl/install-manual):
				1) powershell을 관리자 권환으로 연다
				2) Linux용 Windows 하위 시스템 옵션 활성화 : 
				dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
				3) virtual machine 기능 사용 : dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
				4) https://docs.microsoft.com/ko-kr/windows/wsl/install-manual 의 4단계에서 최신 패키지를 다운로드
				5) WSL2를 기본 버전으로 설정 : wsl --set-default-version 2
				6) Microsoft Store에서 원하는 버전의 Linux을 다운 받는다
			
	2. Anaconda 설치
		a. https://www.anaconda.com/products/individual 에서 다운로드해서 설치
		b. 가상환경을 사용하면 패키지 버전별 독립된 환경을 구현할 수 있습니다. 특히 tensorflow gpu가 요구하는 python, CUDA, cuDNN 버전이 다르기때문에 가상환경에서 개발하는 것을 추천드립니다
		c. 가상환경 생성 : conda create -n [가상환경 이름] python='버전'
		d. 가상환경 활성화 : conda activate [가상환경 이름]
		e. 가상환경 비활성화 : conda deactivate
		f. 가상환경에 깔린 패키지 확인 : cond alist
		g. 다운로드 해야 할 패키지들 : 가상 환경안에서 설치하면 가상환경 안에서만, 밖(base)에서 설치하면 밖에 설치됩니다.
			i. tensorflow, CUDA, cuDNN : ray에서 gpu를 인식하려면 먼저 tensorflow가 gpu을 인식하는지를 확인해야합니다. 
				1) 터미널에 nvidia-smi 명령어를 입력하고 엔터를 누릅니다.
				2) 밑 사진에서 CUDA version은 설치된 CUDA version이 아닌 추천 버전 입니다.
					a) ![image](https://user-images.githubusercontent.com/58590260/148860923-622240ca-8e8b-44e1-9681-2e2cf55c1bde.png)
				3) CUDA version을 확인합니다 : nvcc --version  (https://d33p.tistory.com/16)
					a) 확인이 안되면 CUDA가 설치되지 않았다는 뜻입니다
				4) cuDNN verison을 확인합니다 : cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 (https://d33p.tistory.com/16)
					a) 확인이 안되면 cuDNN이 설치되지 않았다는 뜻입니다
				5) https://www.tensorflow.org/install/source_windows?hl=ko 에서 tensorflow, CUDA, cuDNN 호환 버전을 확인해본다!
				6) tf_detect_gpu.py을 실행하여 기기 목록에 gpu가 뜨는지 확인한다
				7) 추가로, tensorflow에 gpu가 할당되는지 확인하고 싶다면, tf_simple_use_gpu.py 을 실행히본다. 
					a) 정상적으로 할당됐다면 GPU:번호 가 출력된다.
					b) 코드를 실행하기 전에 터미널 창에 nvidia-smi -l 1 을 입력하여 1초마다 nvidia 상태창을 띄워, 실시간으로 GPU 메모리 할당량을 볼 수 있다.
			ii. ray[rlib] : ray 강화학습 라이브러리
	3. SUMO
		a. SUMO 설치하기
			i. 윈도우 : https://sumo.dlr.de/docs/Downloads.php 에서 다운로드해서 설치합니다
			ii. 리눅스
				1) 저장소에 있는 패키지 업데이트 : sudo add-apt-repository ppa:sumo/stable
				2) 설치된 패키지들을 최신 버전으로 업데이트 : sudo apt-get update
				3) sumo 설치 : sudo apt-get install sumo sumo-tools sumo-doc
		b. 환경변수 설정(https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home)
			i. 윈도우 : .msi로 설치하면 자동으로 환경변수가 추가됩니다.
				1) 사용자 환경 변수를 추가합니다
				2) 변수 이름 : SUMO_HOME
				3) 변수 값 : sumo 주소
			ii. 리눅스 : export 명령어를 사용하여 환경변수에 값을 설정할 수 있습니다. 그런데 재부팅할 때마다 환경 변수 값이 사라지기 때문에 로그인할 매마다 실행되는 .bashrc 파일에 export 명령어을 기입합니다.
				1) vi 에디터를 사용 : vi ~/.basrhc
				2) 편집모드 들어가서 수정 : i를 누르고 export SUMO_HOME="SUMO 위치"
				3) 명령모드로 들어가서 저장후 강제 종료 저장 : esc를 누르고 :wq! 입력 
				4) .bashrc 실행 : source ~/.bashrc
	4. 원격과 관련된 프로그램 설치 : Putty, Xming, MobaXterm(Xming 필요없음)
	5. 
		
			
	

