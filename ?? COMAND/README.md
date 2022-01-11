	# 자주쓰는 리눅스 명령어
- sudo [명령어]: 관리자 권한. 명령어를 실행했을 때 루트 어쩌고 저쩌고 하면 sudo를 붙이면 된다. 관리자 권한이니까 조심해서 사용!!!
- su [사용자 권환]: 사용자 권한 변경. su root을 입력하면 관리자 권한 계정으로 들어간다
- cd [위치] : 디렉토리 이동 
- ls : 디렉토리. 파일 목록
- mkdir : 디렉토리 생성
- rm : 파일 삭제 rmdir : 디렉토리 삭제
- pwd  : 현재 위치
- nautilus [위치] : [위치]의 탐색기를 엽니다. nautilus . 이면 현재 폴더의 탐색기를 연다. 서버 컴퓨터에 있는 결과 파일을 터미널 창으로 보기 힘들 때 사용한다.
- vi : 편집 에디터(https://blockdmask.tistory.com/25)
  - ![image](https://user-images.githubusercontent.com/58590260/148896179-4a3db9d2-a38a-46fd-bf5f-cb4bf68e3c49.png)
  -  https://blockdmask.tistory.com/25 에 vi 명령어가 정리되있으니, 참고하여 사용하시면 됩니다.
  - 제가 주로 사용하는 명령어는 i(입력모드), wq!, q! 입니다. 작성하고 싶을 때 i을 눌러 편집 모드로 들어가서 작성하고, 저장할 때에는 esc(명령모드)->:(행모드)->wq!(저장 후 강제로 나감)을 외워서 씁니다. 만약 저장할 필요가 없다면 wq!에서 w을 제외하고 q!만 입력하고 나가면 됩니다.
  - vi에서 코드도 작성할 수 있습니다.
  - 방향키 누를 때마다 abcd 나오는 오류 해결하기
    - 첫번째 방법(https://yang1650.tistory.com/111)
      1) .exerc 파일 생성 : vi ~/.exrc
      2) 편집 모드로 들어가 수정 : i를 누르고 밑의 내용 추가 
      set autoindent
      set bs=2
      set nocp
      3) 명령 모드로 들어가 저장 후 강제 종료  : esc 누르고 :wq! 입력하고 엔터
      4) .exerc 실행 : source ~/.exerc
    - 두번째 방법(https://ledgku.tistory.com/23)
      1) 설치된 패키지 최신 업데이트 : sudo apt-get update
      2)  vim 다운로드 : sudo apt-get install vim
      3) vim 설정파일 수정 : vim ~/.vimrc
      4) 편집모드로 들어가 아래 내용을 복붙한다. 아래 내용대로 설정하면 터미널에서 텍스트보기 편해진다
        a) 
      5) 명령모드로 들어가 저장하고 강제 종료한다 : esc를 누르고 wq!을 입력하고 엔터한다.
      6) 혹시 적용 안됐으면 source ~/.vimrc 시도 해본다!
- cat [파일 이름]: 텍스트 파일 모두 출력
- more [파일 이름]: 텍스트 파일을 한페이지씩 출력
- export [환경변수=값]: 환경변수 설정. 재부팅하면 사라집니다. 그래서 재부팅하면 실행되는 .bashrc 파일에 명령어를 추가합니다. (SUMO 환경변수 설정에 있습니다)
- echo [문자열] 혹은 echo $[변수명] : 문자나 변수 출력. C언어에서 printf와 비슷합니다. 
