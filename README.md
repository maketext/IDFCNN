(다음주 까지 그림과 소스코드를 추가할 예정입니다.)

# I'm Different From you CNN 

## 양품, 비품 판정 사진

- 이차전지 버스바 부품 모형 불량 판정 모습
![최종-bad](https://user-images.githubusercontent.com/32004044/170803342-fb49e4b5-73ab-411a-934b-e8f1203585bf.png)

- 시멘틱 세그먼테이션으로 타겟 Crop 후 5x5 패치별 판정 모습
- 각 패치는 56x56
![라네즈](https://user-images.githubusercontent.com/32004044/170803493-060366f2-7193-428d-b08b-85c2b46a2ffa.png)

## DNN 파이프라인

![Frame 1](https://user-images.githubusercontent.com/32004044/170803562-7fc4b895-c7eb-430c-bb6c-df5b876d3cff.png)

## 모델 설계 방향

- 모델에 원본 이미지를 지속적으로 주입

  - 모델의 초기 부분에 원본 이미지 값을 추가적으로 주입하고 뉴런 간 밀집도를 Sparse 하게 구성

  - 이후 원본 이미지를 메인 모델에 지속적으로 주입
  
  - 모델 마지막 부분에도 원본 이미지 주입

  ![원본 이미지 주입](https://user-images.githubusercontent.com/32004044/170392801-e54c71a8-4750-4d17-985e-58c5970fc755.jpg)
  ![원본 이미지 주입 2](https://user-images.githubusercontent.com/32004044/170392795-76002a87-6d6b-4c8f-98ce-901912fff8ff.jpg)

  - 비전 트랜스포머의 기본 아이디어에서 영감을 얻음
  
  - XY 평면 차원수를 맞추기 위해 단순 0.x 업 샘플링 (바이큐빅 인터폴레이션)을 적용
  
    - 바이큐빅 인터폴레이션은 바이리니어 인터폴레이션 보다 좀 더 넓은 영역의 정보를 취합하기 때문
  
  - 채널 방향으로 CAT 하는 형식으로 주입
  
- 메인스트림과 서브스트림 분화
  
  - 인셉션 모듈 개념 모방
  
  - 가우시안 개념이 적용된 서브스트림을 구성 후 메인 스트림에 통합
  
  - NN 또는 행렬 곱 개념이 적용된 Densely 서브스트림을 구성 후 메인 스트림에 통합
  
  ![병렬스트림](https://user-images.githubusercontent.com/32004044/170392920-e9897a89-e6ae-43ea-a985-8217e7c08d84.jpg)

- 퓨전시 CAT 방식으로 적용

- 논 로컬 피처 추출을 위해 고스트넷에서 유래한 블록 사용, 행렬 곱 사용 (GCNet 참고), NN (FC 층) 사용

- MP 대신 AvgPool 사용
  
  - 평균 값이 최대 값에 비해 정보의 소실이 좀 더 적다고 볼 수 있기 때문

- 모델의 마지막 부분은 Conv 1x1을 연속적으로 쌓아 형성
  
  - 뉴런 값들을 논-로컬 적으로 섞어주면서 차원 축소를 행하는데 연산 부담은 FC에 비해 적기 때문

- 일반 Conv와 Dilated Conv, Group Conv를 적절히 섞음
  
  - 일반 Conv는 상대적으로 적게 사용

## 실전 모델 설계시 생각한 기본 이론

기본 블록 설계부터 해보겠습니다. 블록은 ResNet의 잔차를 사용하지만 구성 레이어 (Conv, BN, ReLU) 의 배치를 바꾸어 보겠습니다.

![컨볼루션 블록](https://user-images.githubusercontent.com/32004044/170392397-30e7e14b-b5cc-4485-a12c-13ec1c107fbe.jpg)

이런 식으로 배치를 바꾸어 각각 테스트를 해볼 수 있습니다. 조금이라도 성능이 좋은 쪽으로 배치를 정하면 될 거 같습니다. 지금은 고전이 된 2015~2017 정도 사이의 논문들을 보면 BN과 활성화 함수를 먼저 배치하고 이후에 컨볼루션 층을 배치해야 성능이 올라간다고 되어 있습니다. 이것은 간단한 실험으로 수긍이 가는데, 입력 이미지의 각 픽셀 값을 0-255 스케일에서 0-10 스케일로 조정을 하여 딥러닝 모델을 돌려보십시오. 훨씬 나은 결과를 기대할 수 있습니다. 이런 실험에서 아마도 BN을 컨볼루션 층 이전에 배치해야 더 나은 결과가 나올 거란 사실을 미루어 짐작 할 수 있습니다.

첫 번째 간단한 모델을 만들어 봅시다.

![모델 설계 1](https://user-images.githubusercontent.com/32004044/170392415-82a1c765-6366-48fc-848d-42e8d48a657c.jpg)

잔차 블록(ResBlock, Residual Block) 2개를 배치하고 바로 FC (Densely Connected Layer) 레이어를 4개 배치해보았습니다. 잔차 블록 내 Conv 3x3 레이어에서 스트라이드 (st) 를 1 또는 2로 줄 수 있습니다. 3 이상도 가능하지만 보통 2를 적용하면 XY 평면 차원 수가 반으로 줄어들어 보통 2나 1을 적용합니다. 이번 모델에서는 잔차 블록을 2개만 사용하기 때문에 평면 차원 수는 줄이지 않을 겁니다. 그래서 스트라이드는 1로 적용합니다. 각 잔차 블록내 Conv 3x3 레이어마다 1로 줍니다. 

Conv 1x1에 스트라이드를 2로 적용할 수도 있습니다. 하지만 보통 Conv 1x1 은 채널 방향으로 사이즈를 줄이거나 늘리는 효과를 가져옵니다. XY 평면 값은 스트라이드 1로 했을 때 변하지 않습니다. 보통 채널 방향으로 축소를 하기 위해 사용합니다. 커널 수를 줄이면 축소가 되겠지요. 그 기능상 차원 축소 즉, 좀 더 의미 있는 정보에 집중하기 위한 어텐션 개념으로 사용합니다. 

저는 Conv 1x1을 인간의 휴식하는 행위에 비유하고 싶습니다. 인간은 공부하다가 잠시 낮잠을 잘 때 그 기억이 정돈되고 고민했던 문제가 조금씩 해결되는 경험을 합니다. 마찬가지로 Conv 1x1 레이어를 여러 번 연속해서 쌓을 경우 낮잠에 비유할 수 있습니다. 저는 그런 비슷한 효과를 가져온다고 믿고 있습니다. 이전에 ICCV 2019 학회에서 중국의 워크샵 포스터를 본 적이 있는데 그때 이미 중국은 Conv 1x1을 잔뜩 연속적으로 이어 붙여 사용하고 있었습니다. 그 당시 1x1이 매우 많이 쌓여 있는 도식도를 보고 충격을 받았었지요.

그럼 잔차 블록을 2개 적용했고, FC 층을 4개 배열합니다. FC층과 층 사이의 뉴런 수는 너무 급격하게 줄어들지 않게 최종적으로 5개 라벨이 정해지도록 잘 조정해 줍니다. 5개의 각 라벨의 의미는 정하기 나름이지만 (x1, y1, x2, y2, Label) 로 하겠습니다.


이번엔 Residual Block을 구성하고 있는 Conv 3x3 레이어에 스트라이드를 2로 항상 적용해 보겠습니다. 그러면 매 잔차 블록을 거칠 때 마다 XY 평면 차원 수가 절반으로 줄어들 것입니다. 첫 번째 Residual Block을 거치고 나면 스트라이드=2 인 Conv 3x3 레이어 때문에 (배치사이즈, 224, 224, 3) 였던 차원 수가 (배치사이즈, 112, 112, 8) 로 되어 XY 평면 차원 수가 절반으로 줄어들었습니다. 총 3개 블록에 스트라이드를 2로 주어 연속으로 3번 차원 축소를 해 보았습니다.

그 결과 (배치 사이즈, 28, 28, 24) 차원 수
에 도달하였고 이를 Flat 화 하여 차원 수를 (배치 사이즈, 28x28x24) 로 2차원으로 조정한 후 NN 층 (FC 층) 을 적용합니다.

![모델 설계 2](https://user-images.githubusercontent.com/32004044/170392428-fccc6482-042e-4866-ba66-57bcf043d6c9.jpg)

## 양품, 비품 데이터셋 생성 방법

```
  {ProjectRoot}/img/원-배치-태스크/불량데이터/0
  {ProjectRoot}/img/원-배치-태스크/불량데이터/1
  {ProjectRoot}/img/원-배치-태스크/불량데이터/2
  {ProjectRoot}/img/원-배치-태스크/불량데이터/3
  {ProjectRoot}/img/원-배치-태스크/불량데이터/배경
```

총 5개의 폴더가 있습니다. 그 중 '0' 폴더에는 양품 패치 이미지 정보를 저장해야 하고 나머지 '1', '2', '3' 폴더에는 각 불량 유형별로 정리하여 저장하면 됩니다. 이미지 저장시 원본 패치인 u1.jpg, u2.jpg... 파일명과 비품을 판정하는 근거가 되는 부분을 굵기 약 10pt인 빨강 (255, 0, 0) 직사각형을 칠한 m1.jpg, m2.jpg... 파일명으로 구분하여 저장합니다. 

패치 이미지 파일은 웹 클라이언트 주소 http://localhost:8888/index 에서 취득합니다. 웹을 구동하기 위해 프로젝트 루트의 "배치.bat" 파일을 실행합니다. 그리고 웹 클라이언트 주소로 접속합니다.

![캡처1](https://user-images.githubusercontent.com/32004044/170846859-0e751de8-2d08-4dde-a3c9-e04e83ec213c.JPG)

"스트리밍 시작"을 클릭하여 이미지를 캡쳐 후 "스트리밍 중지" 버튼을 클릭합니다. 그리고 "판정" 버튼 클릭하면 원본 이미지가 패치화 되는데 이때 model2BboxYOLO.dat 파일이 꼭 프로젝트 루트에 있어야 합니다. DAT 파일은 학습된 뉴런 가중치 값들을 저장하고 있는 pytorch용 파일입니다. 클라우드로 공유합니다.

|OneDrive|
|--------|
|https://manystallingscom-my.sharepoint.com/:u:/g/personal/maketext_manystallings_com/EZUosP2KamtIiUEAEGKoBUkBO1Hikda9Y4gS2LSykqwvkg?e=gA02IM|

원본 이미지가 패치화 되면 18개의 패치이미지가 이 경로상에 생성됩니다.

```
  {ProjectRoot}/img/원-배치-태스크/유효패치-판정
```

'\[idx0]\[0rad]0.0.jpg', '\[idx0]\[0rad]0.1.jpg'... 이미지들을 데이터셋으로 활용하시면 됩니다.

이때 웹캠은 가급적 가로-세로 비가 정사각형에 가까운 큐센 QSENN QC4K 웹캠 http://prod.danawa.com/info/?pcode=12332438 이나 해상도 2048x2448 등의 정사각 에스펙트 레이시오와 비슷한 머신비전 카메라 (오므론, 필라 등) 를 사용하시면 좋겠습니다.

## 학습 방법

"가중치 초기화" 버튼을 먼저 누릅니다.

![캡처2](https://user-images.githubusercontent.com/32004044/170846869-ee140296-237e-4b46-a9b3-bdafe1f7f8a5.JPG)

"생성완료. 학습을 시작하세요." 문구가 창에 뜨면 "학습시작" 버튼을 눌러 학습을 시작합니다. 

![캡처3](https://user-images.githubusercontent.com/32004044/170846873-a02dd5b6-7f6b-4de5-bd0d-4b677824d376.JPG)

학습 도중 "일시중지" 버튼을 한 번 클릭하면 언제든 멈출 수 있고 한 번 더 클릭하면 재개할 수도 있습니다.

![캡처3-1](https://user-images.githubusercontent.com/32004044/170846875-590c25da-941a-48fc-b543-efb6907e1f32.JPG)

학습이 잘 진행되었다고 판단이 되면 "학습결과를 model2BboxYOLO.dat 파일에 저장" 버튼을 클릭합니다. 학습이 끝났습니다. 

## License
IDFCNN is released under the Apache 2.0 license. Please see the LICENSE file for more information.

Copyright (c) Many Stallings. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmaketext%2FIDFCNN&count_bg=%233D76C8&title_bg=%23000000&icon=&icon_color=%23000000&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

