---
title: "Weisfeiler-Lehman Neural Machine for Link Prediction 논문 리뷰"
tags:
  - Graph
  - Neural Networks
  - Link Prediction
  - Weisfeiler-Lehman
use_math: true
---
test

본 포스트는 그래프의 Link prediction 중 휴리스틱을 이용하지 않고 subgraph를 입력으로 Neural Net을 통해 학습시킨 논문(Muhan Zhang and Yixin Chen. 2017. Weisfeiler-Lehman Neural Machine for Link Prediction. In Proceedings of KDD ’17, Halifax, NS, Canada, August 13-17, 2017, 9 pages., [원본 논문](https://dl.acm.org/doi/10.1145/3097983.3097996)) 의 내용을 한국어로 옮긴 것이다. 의역이 상당히 많고 용어에 대한 해석에 대해 한계가 있을 수 있다.



## 1. Introduction

Link Prediction은 친구 추천, 상품 추천, 지식 그래프 구축 등 여러 활용분야에 쓰이며 관심을 얻어왔다. 이를 위해 많은 규칙 기반 모델이나 확률적 matrix factorization 등 모델들이 발전했다. Common neighbor 기법은 두 대상 노드가 공통된 이웃 노드들을 가지고 있을 때 둘 사이에 링크가 존재할 확률이 높다는 직관적이고 간단한 휴리스틱을 가지나 소셜 네트워크와 같은 분야에서 좋은 성능을 냈다. 하지만 이러한 휴리스틱의 문제점은 이와 다른 네트워크에 적용하기에 한계가 있는 범용적 활용도를 가지지 못한다. 한 서베이 논문은 20개의 다른 휴리스틱을 비교분석 했는데 모든 네트워크에 적용 가능한 범용 모델은 없었다. 

그렇다면 대상 네트워크 자체에서 적절한 휴리스틱을 배울 수 있는 모델을 찾아야 한다. 이는 네트워크의 휴리스틱은 네트워크 토폴로지(위상)에서 추출되기에 가능하다. 각 링크의 로컬 패턴을 찾아냄으로써 어떠한 패턴이 어떤 링크를 유도하는지 배울 수 있다는 것이다. 이런 방법에선 다양한 휴리스틱이 로컬 패턴으로 인해 심어지기 때문에 명시적으로 휴리스틱을 정할 필요가 없다. 또한 현존하는 휴리스틱을 적용할 수 없는 네트워크에서도 적절한 휴리스틱을 적용할 수 있다.

이러한 관점에서 논문은 Weisfeiler-Lehman Neural Machine(WLNM)을 제시했는데 작동 순서는 다음과 같다.

1. 각 타겟 링크에 대해 대상의 이웃 관계에서 부분그래프를 뽑아낸다. 이를 링크에 대한 enclosing subgraph라고 한다.
2. 뽑아낸 enclosing subgraph의 인접행렬을 만든다.
3. Link Prediction 모델을 학습하기 위해 Neural Network가 만들어진 인접행렬들에 대해 train한다.

Weisfeiler-Lehman(WL) 알고리즘은 그래프 위상에 기반한 버텍스 순서(vertex ordering)를 정하는 것으로 그래프 라벨링 기법의 한 종류이다. 본 논문에선 기본 WL 알고리즘을 보완한 Palette-WL알고리즘을 사용하여 기존 알고리즘이 가진 계산적 비효율성을 완화하고 enclosing subgraph 표현에 더욱 적합하게 대응한다. 위의 방법을 통해 뽑힌 enclosing subgraph를 신경망을 통해 학습시켜 기존의 휴리스틱보다 뛰어난 성능을 보이며 기존의 휴리스틱을 적용하지 못한 데이터셋에 적용을 가능케한다.



![fig1 overall process](https://user-images.githubusercontent.com/52984138/87409418-dfc84d80-c5ff-11ea-8267-ff726a28e2f6.png)

<center>Fig 1. 주어진 네트워크에 대해 모델은 positive link 샘플(A-B)와 negative link 샘플(C-D)을 트레이닝 링크로 선정해 enclosing subgraph를 뽑아낸다. 다음으로 그래프 라벨링 기법이 사용돼 vertex ordering을 정하고 인접행렬을 만든다. 이렇게 만들어진 인접행렬을 모델에 넣어 학습시켜 fitting을 진행한다.</center>



## 2. Preliminaries



#### 2.1 Link Prediction을 위한 휴리스틱들

기존의 link prediction에 쓰인 간단하면서도 효과적인 휴리스틱들은 노드 사이의 유사도(similarity)를 휴리스틱 방법으로 계산해 링크에 대한 스코어로 사용했다. 기존의 휴리스틱들을 스코어를 구하기 위한 고려하는 이웃 노드들의 최대 hop을 통해 종류를 구별할 수 있다. 

예를 들면 common neighbors(CN)와 preferential attachment(PA)는 두 대상 노드(target nodes; 둘 사이에 링크가 있는지 현재 대상이 되는 노드들)와 1 hop 거리에 있는 이웃들만 고려하기에 first-order 휴리스틱으로 볼 수 있으며 Adamic-Adar(AA)와 resource allocation(RA)는 대상 노드들로부터 최대 2 hop까지의 이웃 노드들을 고려하기에 second-order 휴리스틱으로 구분된다. 같은 논리로 h-order 휴리스틱은 대상 노드로부터 최대 h-hop에 있는 노드들을 고려하며 여기엔 네트워크 전체를 고려하는 high-order 휴리스틱들이 있다. 여기에 들어가는 휴리스틱들은 Katz, rooted PageRank(PR), SimRank(SP)가 있다.



<img src="https://user-images.githubusercontent.com/52984138/88142888-3d2c5200-cc31-11ea-9d66-a6a071fb0eca.png" style="zoom:50%;" />

<center> Fig2. Link prediction에 사용되는 휴리스틱들</center>



#### 2.2 그래프

보통의 그래프 표기법과 같다. $G=(V,E)$  이며 $V={v_1,…,v_n }$이고 $E\in V\times V$이다. $A$는 인접행렬로

$$ A_{(i,j)}=\begin{cases} 1 \quad if \quad (i,j)\in E \\ 0 {} \quad \textit{otherwise}\ \end{cases}$$

이며 $G$가 무방향 그래프라면 $A$는 symmetric하지만 방향 그래프라면 $A_(i,j)\ne A_(j,i)$이다. 버텍스 $x$의 이웃버텍스 집합을 표기하기 위해 $\Gamma^d(x)$를 $d$-hop 거리에 있는, 즉 $x$와의 거리가 $d$보다 작거나 같은 버텍스의 집합으로 한다.



#### 2.3 Weisfeiler-Lehman 알고리즘

그래프 라벨링은 mapping 함수 $l\colon V\to C$로 버텍스 집합을 순서 집합으로 변환하는 과정이다. $C$는 color라 칭하며 만약 단사함수라면 집합 $C$를 인접행렬 에서의 고유한 ordering에 사용할 수 있다.
논문에서 사용된 WL 알고리즘은 PALETTE WL 그래프 라벨링으로 Algorithm 1인 기본 1차원 WL 알고리즘을 기본으로 한 것이다. 이 알고리즘은 색이 바뀌지 않을 때까지(수렴할 때 까지) 반복적으로 버텍스의 색을 업데이트하기에 color refinement 알고리즘으로 분류된다.

WL 알고리즘의 기본 방향은 

1. 대상 버텍스의 이웃 버텍스의 라벨을 이어붙여 대상의 라벨을 augment하고
2. 다른 라벨 종류에 대해 상이한 $c$를 부여한다.
3. 이 과정을 수렴, 즉 $C$가 바뀌지 않을 때까지 반복한다.

자세히 서술하자면 각 버텍스는 초기에 모두 같은 라벨을 부여받고 step 1에서 자신의 1-hop 이웃들의 라벨을 이어 붙여 signature string을 만든다. 모든 버텍스의 signature string을 오름차순으로 정렬하고 차례대로 새 라벨을 부여받는다. 이 때 같은 라벨들엔 같은 값을 부여한다. 이 과정을 버텍스들의 color가 바뀌지 않을 때까지 반복한다.

![PALETTE WL 알고르즘 예시](C:\Users\user\Desktop\실내\GNN,GCN\Link Prediction\Weisfeiler-Lehman Neural Machine for Link Prediction\fig3 WL algo.png)

<center> Fig3. PALETTE WL 알고리즘 예시</center>

WL알고리즘의 장점은 마지막 color가 각 버텍스의 그래프 내에서 구조적 역할(structural role)을 나타내며 상대적 순서를 정의할 수 있는 것이다. 이러한 버텍스의 상대적 순서는 상이한 그래프 사이에서도 비슷하게 나타난다. 이 상대적 순서는 graph CNN에서 conv 필터가 움직이는데 필요한 순차적 순서를 제공해 receptive field를 정하는 데 도움을 주기도 한다.



## 3. Weisfeiler-Lehman Neural Machine

WLNM은 encoded subgraph 패턴과 합쳐진 뉴럴넷 모델이다. 이는 자동으로 위상구조를 학습하여 각 링크의 subgraph 패턴을 통해 새로운 링크를 만들어준다. 우선 subgraph 패턴을 인코딩하기 위해 PALETTE-WL을 사용한다. 앞서 살펴본 바와 같이 이 알고리즘을 통해 각 버텍스의 구조적 역할을 기반으로 한 라벨링을 시행함과 동시에 link prediction에서 중요한 요소인 대상노드와의 거리를 통해 정의된 각 버텍스의 초기 상대적 순서를 유지하기에 유용하다. WLNM은 다음과 같은 세 주요 단계로 구성된다.

1. Enclosing subgraph extraction: 대상 링크들의 K-vertex 이웃 서브그래프를 생성한다.
2. Subgraph pattern encoding: 각 서브 그래프를 PALETTE-WL 알고리즘으로 ordering된 인접행렬로 나타낸다.
3. Neural Net training: link prediction을 위한 비선형의 그래프 위상구조를 학습한다.



#### 3.1 Enclosing subgraph extraction

Enclosing subgraph는 링크의 주변환경을 표현한 것으로 위상정보를 가지기에 링크가 존재할 만 한지 알려준다. 또한 마지막 단계인 NN에 (subgraph, link)의 쌍으로서 입력으로 들어간다. 주어진 어느 한 링크에 대해 Enclosing subgraph는 그 링크의 인근정보(neighborhood) 안에 있는 subgraph이다. 인근정보의 크기는 subgraph 안에 있는 버텍스들의 수로 표기되며 이는 사용자가 정하는 $K$ 값으로 정해진다.

버텍스 x와 y 사이의 링크가 있을 때, 우선 각 버텍스의 1-hop 이웃 버텍스들인 $\Gamma(x)$와 $\Gamma(y)$를 순서 노드 리스트인 $V_K$에 넣는다. 그리고 순차적으로 $\Gamma^2(x)$, $\Gamma^2(y)$, $\Gamma^3(x)$, $\Gamma^3(y), ...,$를 $\|V_K\| \geq K$이거나 남은 이웃이 없을 때까지 반복적으로 추가한다. 모든 Enclosing subgraph는 그 수가 $K$로 동일해야 하기에 만약 $\|V_K\| > K$이라면 $V_K$에 그래프 라벨링을 시행해 순서를 바꾸고 마지막 $\|V_K\| - K$를 빼준다. $\|V_K\| < K$라면 $K - \|V_K\|$개의 더미 노드를 더해준다.

만약 $K \geq \|\Gamma (x)\cup \Gamma (y)\cup x \cup y\|$라면 Enclosing subgraph는 앞서 2-1에서 서술한 first-order 휴리스틱을 적용하기 위한 모든 정보를 담고 있다고 볼 수 있다. 마찬가지로 $K \geq \|\Gamma (x)\cup \Gamma(y)\Gamma^2(x)\cup\Gamma^2(y)\cup x \cup y\|$라면 second-order 휴리스틱을 위한 것이고, 만약 $\|V\|=K$라면 Enclosing subgraph는 모든 high-order를 다룰 수 있다. 이러한 이유로 WLNM이 다른 휴리스틱들 보다 뛰어난 성능을 보이는 것이다.



#### 3.2 Subgraph pattern encoding

Subgraph pattern encoding은 각 Enclosing subgraph를 특정한 버텍스 ordering을 적용한 인접행렬로 표현해 NN의 입력으로 주기 위한 과정이다. Figure 4는 Subgraph pattern encoding의 전체과정을 보여준다.



![img4](C:\Users\user\Desktop\실내\GNN,GCN\Link Prediction\Weisfeiler-Lehman Neural Machine for Link Prediction\fig4 Subgraph pattern encoding.png)

<center> Fig4. WLNM의 실행 과정: 최좌측부터 enclosing subgraph를 뽑아낸 후 subgraph 패턴을 인코딩하고(중간 3단계), 뉴럴넷을 이용해 학습을 진행한다.</center>

##### 3.2.1 PALLETE-WL for vertex ordering

그래프 라벨링에 의해 생성되는 버텍스 ordering은 서로 상이한 그래프라도 일관돼야 한다. 버텍스들은 그들의 위치와 구조적 역할이 비슷할 경우 비슷한 rank(색, 숫자)부여 받아야 한다는 뜻이다.

그래프 라벨링에 필요한 두 가지 조건은 다음과 같다.

1. 두개의 버텍스가 각각의 Enclosing subgraph 내에서 비슷한 구조적 역할을 한다면 비슷한 rank를 부여받아야 한다.
2. Enclosing subgraph내에서 대상 링크를 구별할 수 있어야 하며 Enclosing subgraph내 위상적 방향성(directionality)을 보존해야 한다.

첫번째 조건은 전통적인 WL 알고리즘으로 확보할 수 있으나 두번째 조건은 보장할 수 없다. 즉 전통적 WL 알고리즘은 대상(중심) 링크를 구별할 수 없다. 두번째 조건이 중요한 이유는 다른 보통의 그래프와 달리 Enclosing subgraph는 고유한 방향성을 갖기 때문이다. 대상 링크를 중심으로 이웃 버텍스들이 대상 링크와의 거리를 기반으로 반복적으로 외곽으로 확장되며 그래프가 구성되기 때문이다. 

그러므로, 좋은 그래프 라벨링 알고리즘은 1) 두 중심 버텍스들은 항상 가장 작은 색을 가져야하며, 2) 중심 링크와 가까운 버텍스는 먼 버텍스보다 더 작은 값을 가져야 한다는 조건을 가져 그래프의 방향성을 투영해야 한다. 이 방향성은 버텍스 ordering을 정의하는 데 매우 중요하며 link prediction에 중요한 역할을 한다.

PALETTE-WL 알고리즘은 위의 두가지 조건을 충족시키는 알고리즘으로 본논문의 핵심 아이디어 중 하나이다. 우선 color-order 보존성에 대한 정의는 다음과 같다.



###### color-order 보존성 두 버텍스 $v_a$와 $v_b$가 있을 때, 어떤 한 반복 $i$에서 $v_a$의 색이 $v_b$의 색보다 작은 숫자라면 $i + 1$에서 또한 $v_a$의 색이 $v_b$의 색보다 작은 숫자여야 한다.



같은 이유로 초기의 $v_a$의 색이 $v_b$의 색보다 작았다면 마지막(final) 색 또한 그 관계를 유지해야 한다. 그러므로 초기에 버텍스의 라벨들이 중심 링크와의 거리를 기준으로 오름차순으로 구성됐다면 color-order 보존성의 정의에 의해 refine 후에도 그 순서를 유지할 것이라는 것이다.

예를 들면 라벨링 초기화 단계에서 대상 링크의 두 버텍스에 색 1을 부여하고 1-hop 이웃에건 색 2, ... n-hop 이웃에겐 색 n+1을 부여한 후 색 refine 알고리즘을 실행하면 Enclosing subgraph의 color-order 보존성에 의해 최종 라벨은 여전히 거리를 기반으로 한 라벨 order를 보존할 것이다. 또한 대상 링크의 두 대상 버텍스인 $A_1$과 $A_2$는 Enclosing subgraph 내에서 가장 작은 두 색을 부여받을 것이다.

기본적인 WL알고리즘은 color-order 보존성을 유지하기에 WLNM을 위한 알맞은 그래프 라벨링 기법이지만 버텍스의 signature string을 생각해보면 연산적으로 매우 비효율적이다. 이에 해싱기반의 WL알고리즘이 제기되어 해시 함수 $h(x)$를 통해 고유한 값에 고유한 signature를 맵핑해준다. 이로인해 버텍스들은 signature string 대신에 이 고유한 해시값을 통해 구분될 수 있다. 이는 전통적인 WL 알고리즘에 비해 매우 빠르다.

버텍스를 위한 해시 함수는 다음과 같다.
$$
h(x) = c(x) + \sum_{z\in \Gamma(x)}\log(P(c(z)))
$$
여기서 $c(x)$는 정수값으로 버텍스 $x$의 color 값을 말하고 $P$는 소수 배열을 뜻한다. $P(c(x))$는 소수배열에서 $c(x)$번째 원소를 반환한다. 위와 같은 해시함수를 통해 각 버텍스의 고유 해시값을 반환 받는데 만약 버텍스 $x, y$에 대해 $h(x) = h(y)$는 $c(x) = c(y)$이고 $\Gamma(x)$와 $\Gamma(y)$가 같은 색들, 관계 카디널리티(같은 새로운 색 = 같은 WL signature)를 만족할 때 가능하다.

연산적, 시간적으로 효율적이나 위의 함수는 color-order 보존성을 갖지 못한다. 즉 구조적 역할에 의해 coloring이 진행되지만 의미있는 순서를 나타내지 못한다는 것이다. 또한 위의 수식에 의한 WL은 수렴하지 못하는 일도 발생한다. 이를 해결하기 위해 PALETTE-WL을 고안했는데 이는 다음과 같다. 
$$
h(x) = c(x) + \frac{1}{\lceil \sum_{z' \in V_k} \log(P(c(z')))\rceil}\cdot \sum_{z\in \Gamma(x)}\log(P(c(z)))
$$
여기서 $\lceil \cdot \rceil$은 ceil 함수를 뜻하며 $V_k$는 Enclosing subgraph의 라벨링될 버텍스 집합을 뜻한다. 

이 모델을 PALETTE\_WL이라고 명명했으며 전체 알고리즘은 알고리즘 1과 같다.

<img src="C:\Users\user\Desktop\실내\GNN,GCN\Link Prediction\Weisfeiler-Lehman Neural Machine for Link Prediction/alg1.png" alt="alg1" style="zoom:85%;" />



여기서 $d(v_a, v_b)$는 버텍스 $v_a$와 $v_b$ 사이의 가장 가까운 경로의 길이를 뜻한다. 버텍스에 라벨, 즉 color를 부여할 땐 함수 맵핑 함수 $f \colon \mathbb{R}^K \to C^K$를 사용해 가장 작은 실수부터 $c1$을, 그다음으로 작은 실수에 $c2$를, ... 이런 식으로 $K$까지 순차적으로 부여한다. 마지막으로 $V_K$에 있는 버텍스들을 오름차순으로 정렬한다.



##### 3.3.2 Represent enclosing subgraphs as adjacency matrices

Enclosing subgraph $G(V_K)$가 주어졌을 때, WLNM은 이것을 버텍스 order가 $V_K$의 PALETTE-WL color로 결정된 상단 삼각형(매트릭스의 대각선(diagonal)을 기준으로 위쪽 부분)로 표현한다. 그 후 Neural Net의 입력으로 들어간다.

본 연구에선 binary 값을 갖는 본래의 인접행렬을 변형해서 $A_{i,j} = 1/d((i,j),(x,y))$로 했는데 이는 대상링크 $(x,y)$에서 링크 $A_{i,j}$까지의 최단경로의 길이를 뜻한다.



#### 3.3 Neural network learning

**Training**: 다음으로 NN을 통해 비선형 패턴을 학습시켜 예측을 시행하는 단계이다. 첫째로 인풋 네트워크(그래프) $G=(V,E)$에서 모든 엣지인 $(x,y) \in E$를 positive sample로 구성한다. 다음으로 negative sample을 만들기 위해 랜덤하게 고른 $\alpha\|E\|$개의 $x,y\in V, (x,y) \notin E$를 구성한다. 이렇게 구성된 트레이닝 셋의 모든 링크에 대해 각각의 Enclosing subgraph를 만든 후 PALETTE-WL을 사용해 인접행렬을 구한다. 이 트레이닝 셋은 라벨과 함께$(1:(x,y)\in E, 0: (x,y) \notin E)$ NN의 인풋으로 들어간다.

이때 $A_{1,2}$는 링크 $(x,y)$를 의미하기에 NN에 인풋으로 들어가면 안된다. 학습 중 이 값은 1 또는 0이 될 수도 있지만 예측시엔 0으로 맞추고 진행한다. 이러한 클래스 라벨을 더한 학습과정을 거쳐 모든 테스팅 링크에 예측값을 부여할 것이다.



**Testing(link prediction)**: 학습을 진행한 후 같은 과정으로 PALETTE-WL를 통해 예측 대상 좌표쌍의 Enclosing subgraph를 구한 뒤 인접행렬을 구축해 NN에 넣은 후 예측을 진행할 수 있다. 결과로 0과 1사이의 값이 나올 것이며 이는 테스트 링크가 얼마나 positive한지에 대한 확률 값을 나타낸다.



## 4. Experimental results

Figure 4는 두개의 인공적인 데이터셋과 실제 데이터인 USAir-network의 그래프와 Enclosing Subgraph들, 그리고 네트워크에서 학습된 가중치행렬을 나타내고 있다. 각 네트워크의 가중치행렬을 비교해보면 상이한 네트워크 사이에서도 비슷하게 형성되는 것을 확인할 수 있다. 또한 PALETTE-WL 알고리즘을 통해 형성된 Eclosing subgraph의 라벨을 보면 대상링크는 항상 color 1과 2를 가지며 다른 라벨들을 그래프 내에서 구조적 역할에 따라 color를 부여받으며 enclosing subgraph의 방향성을 유지하는 것을 확인할 수 있다.

![fig4](C:\Users\user\Desktop\실내\GNN,GCN\Link Prediction\Weisfeiler-Lehman Neural Machine for Link Prediction\fig 5 visualization.PNG)

<center> Fig 4. 좌측: 3-regular graph, 가운데: preferential attachment graph, 우측: USAir network</center>



다음으로 다양한 실제 데이터셋을 사용한 다른 휴리스틱과의 성능, 퍼포먼스 비교이다. Figure 5는 그 결과를 보여주는데 거의 모든 부분에서 WLNM이 다른 휴리스틱보다 좋은 성능을 보이며 특히 Power와 Router는 다른 휴리스틱에서 볼 수 없는 뛰어난 성능을 보여준다.

![fig4](C:\Users\user\Desktop\실내\GNN,GCN\Link Prediction\Weisfeiler-Lehman Neural Machine for Link Prediction\fig 6 results comparison.PNG)

<center> Fig 5. 다양한 데이터셋에 대한 다른 휴리스틱과의 비교(AUC)</center>

$\textrm{WLNM}^K$에서 $K$는 Enclosing subgraph를 구축하기 위한 버텍스의 개수를 의미한다. 신경망 학습을 위하여 NN의 하이퍼 파라미터로 32, 32, 16차원의 3-fully connected layers를 사용했으며 출력 레이어에선 softmax를 사용했다. 모든 은닉층에서 활성화 함수로 ReLU를 적용했으며 역전파를 통한 업데이트를 위해 optimizer로 Adam optimizer에 초기 learning rate로 1e-3을 사용했다. mini-batch의 사이즈는 128이고 학습을 위한 총 epoch는 100을 적용했다. 트레이닝에 사용된 데이터의 비율은 10\%이고 나머지는 validation과 test셋으로 사용됐다. NN과의 비교를 위해 선형회귀 모델인 WLLR을 추가했으며 $K$를 10으로 설정했다.

실험결과를 통해 WLNM은 이전의 휴리스틱 알고리즘이 찾을 수 없는 새로운 위상적 특징을 찾을 수 있다는 것을 확인할 수 있으며 $K$의 숫자에 큰 상관 없이 좋은 성능을 보인다.



## 5. Conclusion

본 논문에서는 link prediction을 위한 새로운 방법으로 Weisfeiler-Lehman Neural Machine을 제시했다. 이 방법에선 각 링크의 local enclosing subgraph를 찾아 그래프 내에서의 위상속성을 구한다. 링크의 enclosing subgraph를 정확히 구하기 위해 효율적인 그래프 라벨링 알고리즘인 PALETTE-WL 알고리즘을 제안했으며 이를 통해 subgraph의 버텍스에 각자의 구조적인 역할을 기준으로 라벨을 부여할 수 있다. 그 후 각 enclosing subgraph의 인접행렬을 구축한 뒤 신경망의 입력으로 줘 학습을 진행했다. 결과는 다른 상용 휴리스틱과 비교해 큰 성능발전을 보였으며 복잡한 네트워크의 위상정보를 배울 수 있고 상이한 모든 네트워크에서 높은 정확도를 보였다.
