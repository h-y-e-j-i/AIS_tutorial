# 02 원격 접속
## 서버컴퓨터의 원격 접속 환경 설정하기
### SSH (원격 접속 프로토콜) 열기 (http://programmingskills.net/archives/315)
1. ssh 설치 : sudo apt-get install openssh-server
2. GUI 관련 패키지 다운로드 (https://blog.naver.com/n_cloudplatform/221451046295)
  1. apt-get install -y ubuntu-desktop xorg xrdp xserver-xorg mesa-utils xauth gdm3 
  2. dpkg-reconfigure xserver-xorg
3. ssh 설정 : sudo vim /etc/ssh/sshd_config
  1. 관련된 옵션은 https://wiseworld.tistory.com/72 에서 보고 필요 없는건 주석 처리하기
      - X11 Forwarding 권한 활성화 : X11Forwarding yes , ForwardX11Trusted yes
      - 서버 컴퓨터 ssh 포트번호는 7777로 설정되어 있다.
  2. ssh 실행 확인 : sudo systemctl status ssh
  3. ssh 실행 : sudo systemctl enable ssh -> sudo systemctl start ssh
  4. 방화벽 ssh 허용 : sudo ufw allow ssh
  5. 방화벽 상태 확인 : sudo ufw status
### FTP 열기
- 실패ㅠㅠ

## 서버 컴퓨터로 원격 접속하기
### 리눅스
1. 터미널에 ssh user@adderss:port [옵션] 형식으로 입력합니다.
  - 예) 서버 컴퓨터에 연결할 때에는 사용자 이름은 sonic, ip는 172.16.111.70, port는 7777, 옵션은 X(X Forwarding 관)이므로 ssh sonic@172.16.111.70:7777 -X
3. 비밀번호를 입력하고 원격 접속합니다
4. 터미널 창에 display 환경 변수를 확인합니다 : echo $DISPLAY
5. dispaly 환경 변수가 정의되지 않았거나 재정의 할 때 : export DISPLAY=클라이언트ip:display number
   - 예) export DISPLAY=172.xxx.xxx.xxx:0
### 윈도우
#### 원격 데스크톱 연결
- 서버 아이피을 입력한 후 사용자 아이디와 비밀번호를 입력하여 데스크탑 모드로 접속한다
- 화면이 안 나올 때 : https://koreatech.tistory.com/entry/%EB%A6%AC%EB%88%85%EC%8A%A4-%EC%9B%90%EA%B2%A9%EC%A0%91%EC%86%8D-xrdp-%EC%84%A4%EC%A0%95%ED%99%94%EB%A9%B4%EC%95%88%EB%82%98%EC%98%AC%EB%95%8C 을 참고하여 해결하기
#### X Window
- 클라이언트가 X server에 요청하면, X server는 클라이언트가 요청한 GUI을 디스플레이 장치 등에 출력한다
- 윈도우는 Xming이라는 X Window 프로그램을 설치해야합니다. 
- 리눅스에서는 기본적으로 제공되서 설치할 필요가 없다. (그런데 제가 사용했던 컴퓨터 기준으로 sumo를 포함한 몇몇 프로그램이 안 띄워지더라구요ㅠㅠ 이유는 모르겠습니다.)
- 실행 방법
  1. Display Number : 모니터 번호가 아니랴 x dispaly 번호 입니다. 출력할 때에 몇 번의 x display에 표시할 지 지정해야 한다. xming을 여러 개 실행한다면 다 다른 숫자를 부여해야 한다
  2. 다음 버튼을 누르다면 추가 파라미터 값을 넣는 칸이 있는데 거기에 꼭!!!! -ac을 적어야 한다
-  PuTTY : 서버 컴퓨터 터미널로 접속할 때 사용하는 프로그램. (https://talkingaboutme.tistory.com/entry/Linux-X11-Forwarding-using-PuTTY)
  1. Host Name와 Port에는 각각 원격에 연결할 서버 IP와 Port을 입력한다
      - 서버 컴퓨터에 연결할 경우 172.16.111.70, Port에는 777을 입력합니다
      - ![image](https://user-images.githubusercontent.com/58590260/148891994-df5fb94e-ad1f-4c59-9d17-bf4ba1390d11.png)
  2. Connection-SSH-X11에서 Enable X11 Forwarding을 체크하고, X dispaly location에 클라이언트ip:displaly number 을 입력합니다.
      - 예) 172.xxx.xxx.xxx:0
      - ![image](https://user-images.githubusercontent.com/58590260/148892162-c105d528-9aae-4ea1-ba32-8f2003f566a1.png)
  3. open을 클릭하고 사용자 이름과 비밀번호를 입력하여 원격접속합니다
  4. 터미널 창에 xclock을 입력해 Xming이 정상적으로 작동하는지 확인합니다
- MobaXterm(X Window가 내장되어 있어 Xming이 필요 없음) (https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=g00dmoney&logNo=20169845760)
  1. Retmoe hostname에 ip을, port는 7777, 그리고 X11 forwarding을 꼭 체크하고 OK 버튼을 클릭한다
    - ![image](https://user-images.githubusercontent.com/58590260/148894372-90ccb779-624c-43fd-9c3a-793615a28841.png)
  2. 사용자 아이디와 비밀번호를 입력하여 접속한다
  3. 터미널에 xclock을 입력하여 X11 forwarding 작동을 확인한다
### 공통
#### VS CODE에서 원격 접속하기
1. 확장 프로그램에서 원격 접속 프로그램을 다운로드한다. WSL을 사용한다면 Remote-WSL, Docker에 리눅스를 깔았다면 Remote-Container, 서버 컴퓨터로 연결하려면 Remote-SSH를 설치한다.
2. Remote-SSH 에 연결했다면 왼쪽 하단에 초록 버튼을 클릭하고 Coneect Host를 클릭한다.
  - ![image](https://user-images.githubusercontent.com/58590260/148892233-ca720892-05ac-4abe-84cc-28f98087d6fe.png)
3. ssh user@adderss:port [옵션] 형식으로 입력합니다.
  - 예) 서버 컴퓨터에 연결할 때에는 사용자 이름은 sonic, ip는 172.16.111.70, port는 7777, 옵션은 X(X Forwarding 관)이므로 ssh sonic@172.16.111.70:7777 -X
4. 사용자 아이디와 비밀번호를 입력하여 원격 접속한다
5. 터미널 창에 display 환경 변수를 확인합니다 : echo $DISPLAY
  - dispaly 환경 변수가 정의되지 않았거나 재정의 할 때 : export DISPLAY=클라이언트ip:display number
    - 예 : export DISPLAY=172.xxx.xxx.xxx:0
6. 윈도우라면 Xming을 실행하고, xclock 명령어를 입력하여 시계가 표시되는지 확인한다
  - 만약 표시되지 않는다면 DISPLAY 환경변수를 바꿔본다(가끔 이상하다;;)
  - 예) export DISPLAY=클라이언트 ip:10.0, export DISPLAY=localhost:10.0, export DISPLAY=localhost:0.0 등등
7. 사용자가 정의한 환경변수는 클라이언트에서 접속이 끊길 때 사라지므로 설정파일을 수정한다 
  ![image](https://user-images.githubusercontent.com/58590260/149275012-6617b2f2-4a72-4ad4-b49c-441a54ed9cb7.png)
    1. 왼쪽 아래 톱니바퀴를 클릭하고 Settings을 클릭한다.
    2. 검색창에 linux을 검색한다
    3. Terminal > Intergrated > Env:Linux 항목에 Edit in settings.json을 클릭한다
    4. 중괄호 안에 아래 내용을 추가한다
    ```json
    "terminal.integrated.env.linux": {
        "DISPLAY" : "클라이언트 ip 혹은 localhost:display number"
    },
    ```
## 개인적으로 사용하는 방법!
- ray dashboard나 tensorboard을 파이어폭스에 주소를 입력해서 접속하는데, putty에서 파이어폭스 접속하면 생각보다 많이 느리다. 그래서 비교적 빠른 원격 데스크톱 연결을 통해 파이어폭스에 접속하여 ray dashboard와 tensorboard를 본다.
- VS CODE는 코드 실행하는 용으로 두고, 추가적인 터미널이 필요할 때에는 putty을 통해 원격 . putty는 여러 창 띄울 수 있다.


