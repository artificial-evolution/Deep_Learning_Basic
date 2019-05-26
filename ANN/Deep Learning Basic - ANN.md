# Deep Learning Basics

## Contents

1. Machine Learning
   1. Concept of Machine Learning
   2. Types of learning algorithms
      1. Supervised Learning
      2. Unsupervised Learning
      3. Reinforcement Learning
2. Deep Learning and Artificial Neural Network
   1. Deep Learning
   2. Artificial Neural Network
   3. Single Layer Perceptron
      1. What is Perceptron?
      2. How does the neuron work?
      3. Perceptron Architecture
         1. Basic Structure
         2. Activation functions
         3. Cost functions / Loss functions
         4. Learning Algorithms
      4. XOR Problem
   4. Multi Layer Perceptron
      1. What is MLP?
      2. MLP Architecture
         1. Basic Structure
         2. Learning Algorithm : Back Propagation
3. Deep Neural Network
   1. Deep Neural Network
   2. Convolution Neural Networks
   3. Recurrent Neural Networks
   4. Long Short-Term Memory models

-----



## 1. Machine Learning

### 1-1. Concept of Machine Learning

**Machine Learning**이란,컴퓨터가 어떤 작업을 수행할 때, 명시적인 지시 없이 패턴과 추론에만 의존하여 알고리즘과 통계적 모델을 학습하고 개발하는 분야를 말한다.

### 1-2. Types of learning algorithms

Machine Learning의 분야는 학습 알고리즘을 기준으로 주로 Supervised Learning(지도 학습), Unsupervised Learning(비지도 학습), Reinforcement Learning(강화 학습)으로 나뉜다. 

#### 1-2-1. Supervised Learning

**지도 학습**이란, 입력 레이블과 레이블에 대한 정답으로 이루어진 학습 데이터를 통하여 해당하는 입력에 대해 정답이 나올 수 있도록 학습한 후, 결과적으로 어떠한 입력에 대해 원하는 정답을 예측할 수 있도록 하는 학습 방법이다.

각 학습 데이터에는 명확한 해답이 사용되며, 정답 레이블이 이산 값이면 Classification(분류), 연속 값이면 Regression(회귀)라고 불린다.

ex) softmax regression

지도학습을 진행할 때는 주로 CNN과 RNN과 같은 ANN 모델이 사용된다.

#### 1-2-2. Unsupervised Learning

**비 지도 학습**이란, 입력 데이터만 주어지고, 입력 데이터들의 숨겨진 특성(hidden feature)이나 구조를 발견하는 방법이다.

입력값에 대한 명확한 해답이 없다는 부분에서 지도학습과 차이를 보인다.

Clustering 등의 알고리즘이 비 지도 학습에 속하며, 주로 Auto Encoder라는 ANN모델이 사용된다.

#### 1-2-3. Reinforcement Learning

**강화 학습**이란, 행동 심리학에서 영향을 받았으며, 현재 상태에서 할 수 있는 행동들이 주어지고, 각 상황에 따른 보상이 주어지는 환경에서 에이전트가 정의되어있을 때, 에이전트가 해당 환경에서 보상을 최대화 하거나, 보상을 최대화 하는 행동 순서를 선택하는 방법이다.

강화학습 모델로 DQN을 많이 사용한다.



## 2. Deep Learning and Artificial Neural Network

### 2-1. Deep Learning

**Deep Learning**이란, 인공 신경망을 바탕으로, 비선형 함수들의 조합을 통해 복잡한 데이터에서 추상화를 시도하는 Machine Learning의 분야중 하나이다.

Deep Learning에는 Deep Neural Networks, Convolutional Neural networks, Recurrent Neural Network 등의 모델이 있다.

### 2-2. Artificial Neural Network

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/300px-Colored_neural_network.svg.png)

**인공 신경망(Artificial Neural Network)**이란, 인간의 신경망에서 영감을 얻은 통계학적 학습 알고리즘으로, 시냅스의 결합으로 네트워크를 형성한 인공 뉴런이 학습을 통해 시냅스의 결합 세기를 변화시켜, 문제 해결 능력을 가지는 모델 전반을 가리킨다.  McCulloch, Warren S.와  Walter Pitts.가 발표한 “A logical calculus of the ideas immanent in nervous activity”에서 최초로 제안되었다.

많은 입력에 의존하며, 이해하기 어려운 구조를 추측하고 근사치를 내는 경우에 사용한다.

일반적으로 다층 퍼셉트론(Multi Layer Perceptron)을 가리키지만, 이에 국한되는 것은 아니다.

### 2-3. Single Layer Perceptron

#### 2-3-1. What is Perceptron?

![img](http://solarisailab.com/wp-content/uploads/2017/05/perceptron_image.png)

 **퍼셉트론(Perceptron)**은 1957년 Frank Rosenblatt가 고안했으며, 인공 신경망을 이루는 인공 뉴런의 모델 중 하나이다.  McCulloch, Warren S.와  Walter Pitts.가 제안한 인공 신경망의 개념을 수학적인 모델로써 나타낸것이다.

#### 2-3-2. Perceptron Architecture

##### 2-3-2-1. Neuron Architecture

![img](http://piramvill2.org/wp/wp-content/uploads/2018/04/%EC%9D%B8%EA%B3%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D_%EB%89%B4%EB%9F%B0_%EC%8B%9C%EB%83%85%EC%8A%A4_%EB%89%B4%EB%9F%B0_%EC%9C%A0%EC%B6%94%EC%98%81%EC%96%B4-632x356.png)

인공 뉴런은 실제 뉴런에서의 몇 가지 특징을 가져온 것이다.

시냅스-전-뉴런의 축삭종말에서 전달되는 신호들의 강도를 입력($x_i $)으로 가져오고,

시냅수-전-뉴런의 축삭종말의 신경전달물질의 양을 가중치($w$)로 가져왔다.

시냅스 후의 값들은 신경 세포체에서 합산되는데, 이는 가중치와 입력값의 곱을 합해주는 과정이다.

그리고 이 값이 활동 전위의 형태로 축삭을 통해 전달되는데, 이를 활성화 함수(a)와 활성값으로 하였다.

이러한 다른 뉴런들의 신호들을 모두 모아 임계치($\theta $)보다 크면 자극을 주고(1), 작으면 자극 주지않는다(0).

##### 2-3-2-2. Basic Structure

퍼셉트론(단층 퍼셉트론)은 출력층과 입력층으로만 이루어져있다.

입력층은 뉴련의 가지돌기와 연결된 다른 뉴런들의 축삭돌기 말단을 모방한 것으로, 다른 뉴런들의 축삭돌기 말단에서 오는 신호들을 입력으로 표현한다.

그리고 뉴런과 다른 뉴런들의 축삭돌기들 말단 사이의 연결 정도가 존재하는데, 이를 가중치로 표현한다.



따라서, 뉴런을 본떠 i번째 입력노드의 입력값을 $x_i$, i번째 입력노드와 연결된 가중치를 $w_i$, 퍼셉트론의 출력을 $y$, 임계값을 $\theta$ 라고 정의하고, $y$를 다음과 같이 수식으로 나타낼 수 있다.
$$
w = (w_1, w_2,..,w_N)
\\
x = (x_1,x_2,...,x_N)
\\
y=\begin{cases} 0, (x_1w_1+x_2w_2+..x_Nw_N\leq \theta) \\
1 , (x_1w_1+x_2w_2+..x_Nw_N>\theta)    \end{cases}
\\
=\begin{cases} 0, (\sum_{i}^{N}{w_ix_i}\leq \theta)\\ 
1 , (\sum_{i}^{N}{w_ix_i}>\theta)   \end{cases}
\\
\therefore y=\begin{cases}  0, (x\cdot w\leq \theta)\\
1 , (x\cdot w>\theta) \end{cases}
$$

##### 2-3-2-3. bias

$y$에 대한 식의 우항을 정리하기 위하여 $b=-\theta$라고 정의하고, $y$는 다음과 같이 나타낼 수 있다.
$$
y=\begin{cases}  0, (x\cdot w +b \leq0)\\
1 , (x\cdot w +b>0) \end{cases}
$$


#### 2-3-3. AND, OR 

퍼셉트론을 이용하여 AND 연산을 수행하는 인공 뉴련과 OR 연산을 수행하는 인공 뉴련을 만들 수 있다.

게이트의 두 입력을 $x_1, x_2$ 라고 할때, 두 가중치를 $w_1,w_2$를  $w_1=1, w_2=1$로 두고, $\theta=-1$로 두면 AND 연산을 만족한다. (편의상 임계값 $\theta$를 사용하여 나타낸다.)
$$
y=\begin{cases}  0, (x_1+x_2\leq1)\\
1 , (x_1+x_2>1) \end{cases}
$$
이번엔 두 가중치를 $w_1,w_2$를  $w_1=2, w_2=2$로 두고, $\theta=3$로 두면 OR연산을 만족한다.
$$
y=\begin{cases}  0, (2x_1+2x_2\leq3)\\1 , (2x_1+2x_2>3) \end{cases}
$$
이와같이 퍼셉트론의 가중치를 조절하면, 퍼셉트론으로 일부 논리 게이트를 만들 수 있다.

#### 2-3-4. XOR Problem

![img](https://cdn-images-1.medium.com/max/1600/1*Tc8UgR_fjI_h0p3y4H9MwA.png)

AND나 OR 같은 간단한 논리게이트들은 쉽게 구현할 수 있다. 

하지만, 퍼셉트론은 단순히 선형 분류기이기 때문에, XOR 연산 같은 경우는 퍼셉트론으로 연산을 정의할 수 없다.

이러한 퍼셉트론의 한계를 수학적으로 증명한 논문이 1969년 Marvin Minsky와 Seymour Papert가 저술한 “Perceptrons: an introduction to computational geometry”라는 논문이다.

이 논문이 한동한 인공지능의 개발을 급격하게 사그라들게 하였다.

하지만, 이러한 XOR 문제를 해결한 것이, 다음에 나오는 **MLP**이다.

### 2-4. Multi Layer Perceptron

![A schematic diagram of a Multi-Layer Perceptron (MLP) neural network.Â ](https://www.researchgate.net/profile/Junita_Mohamad-Saleh/publication/257071174/figure/fig3/AS:297526545666050@1447947264431/A-schematic-diagram-of-a-Multi-Layer-Perceptron-MLP-neural-network.png)

#### 2-4-1. Concept of MLP

인공신경망은 여러개의 계층으로 이루어져있는데, 가장 첫번째 계층을 입력 계층, 가장 마지막 계층을 출력계층이라고 한다. 그리고 입력 계층과 출력계층을 제외한 나머지 계층들을 **은닉 계층**이라고 하는데, 단층 퍼셉트론을 쌓아올려 은닉 계층이 존재하는 퍼셉트론을 **다층 퍼셉트론(Multi Layer Perceptron)**이라고 한다.

#### 2-4-2. MLP Architecture

##### 2-4-2-1. Basic Structure

다층 퍼셉트론과 단층 퍼셉트론의 차이는, 다층 퍼셉트론에는 은닉 계층이 존재한다는 것이다.

단층 퍼셉트론에서는 임계값을 기준으로 값이 바뀌게 되는데, 특정 값을 기준으로 0이나 1의 값을 갖는 함수를 **계단 함수**라고 하고, 이러한 함수를 **활성화 함수**라고 한다.

이를 이용하여 활성화 함수(a)를 계단 함수로 정의할떄, 단층 퍼셉트론은 다음과 같이 표현할 수 있다.
$$
a(x)=\begin{cases}  0, (x \leq0)\\
1 , (x>0) \end{cases}
\\
\therefore y = a(w\cdot x +b)
$$
이것이 다층 퍼셉트론과, 다층 퍼셉트론의 각 노드의 기본 구조이다.



표기법을 한번 정리해보면, 다음과 같은 표기 규칙을 따른다.

| 표기         | 설명                                                         |
| ------------ | ------------------------------------------------------------ |
| $x_i  $      | i번째 노드의 입력 ($=z_i^1$)                                 |
| $w_{i\,j}^k$ | k-i번째 레이어의 i번째 노드에서 k번째 레이어의 j번째 노드로 가는 간선의 가중치 |
| $w^i_j$      | i번째 레이어의 j번째 노드로 가는 간선의 가중치 벡터<br />$w^i_j = \begin{bmatrix} w_{1\,j}^i \\ w_{2\,j}^i \\ w_{3\,j}^i \\ :\end{bmatrix}$ |
| $w^i$        | i번째 레이어의 가중치 행렬, $w^i=   \begin{pmatrix}  w_{1\,1} ^i& w_{1\,2} ^i& w_{1\,3}^i & .. \\  w_{2\,1} ^i& w_{2\,2} ^i& w_{2\,3}^i & ..  \\ w_{3\,1}^i & w_{3\,2}^i & w_{3\,3} ^i& .. \\  : & : & :  \end{pmatrix}$ |
| $b_{i}^{j}$  | j번째 레이어의 i번째 노드의 편향(임계치)                     |
| $a_{i}^{j}$  | j번째 레이어의 i번째 노드의 활성화 함수                      |
| $x$          | 입력 벡터, $x = (x_1,x_2,...,x_N)$                           |
| $z_{i}^{j}$  | j번째 레이어의 i번째 노드의 활성값, $z_{i}^{j} = a_{i}^{j}(w_i \cdot z^{j-1} +b^j_i)$ |
| $z^{i}$      | $z^{i} = ( z_1^i,z_2^i,z_3^i,..,z_N^i) $                     |

##### 2-4-2-2. Activation functions

**활성화 함수**에 대하여 다시 설명하자면, 현재 층에서 다음 층으로 출력값을 넘겨줄 때, 이 출력값의 형태를 변환해주는 함수를 말한다.

활성화 함수로는 비선형 함수를 쓰는데, 그 이유는 선형 함수는 가중치와 편향의 학습으로 대체 가능하므로, 층을 깊게 하는 의미가 없어지기 때문이다.

이 이외에도 활성화 함수로 **Sigmoid 함수**나, **tanh 함수 (hyperbolic tangent**) , **ReLU 함수(Rectified Linear Unit)**를 사용한다.
$$
sigmoid(x)=\frac{1}{1+e^{-x}}
\\
tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
\\
ReLU(x)=\max(0,x)
$$

#### 2-4-3. Learning Algorithm

##### 2-4-3-1. Cost functions / Loss functions

인공신경망을 학습시키려면, 어떤 데이터에 대해 인공 신경망이 추론한 결과와 실제 정답을 비교하여, 실제 정답과 비슷하게 추론하게 해야한다. 여기서, **비용 함수(Cost function) 또는 손실 함수(Loss function)**를 정답과 추론 결과의 유사한 정도의 지표로 사용하면, 인공신경망의 학습이라는 것이 cost function을 최소화 또는 최대화 하는 최적화 문제(Optimization problem)로 바뀐다.

비용 함수로 **평균 제곱 오차(Mean Square Error)**, **교차 엔트로피 오차(Cross Entropy Error) **가 주로 사용된다.

$E$를 손실 함수의 값이라고 했을 때, 평균 제곱 오차의 값은 다음과 같다.
$$
MSE(y,t)=\frac{1}{2}\sum_k(t_k-y_k)^2
$$
교차 엔트로피 오차의 값은 다음과 같다.
$$
CEE(y,t)=-\sum_kt_k\log y_k
$$
두 함수는 말 그대로 오차이기 때문에, 최소화 할수록 정확도가 올라가며 학습이 되게된다.

정확도는 손실함수로써 사용할 수 없는데, 이는 각 test set마다 손실 함수의 값이 달라지기 때문이다.

##### 2-4-3-2. Gradient Descent 

![gradient descentì ëí ì´ë¯¸ì§ ê²ìê²°ê³¼](https://t1.daumcdn.net/cfile/tistory/25343A4B534CD49A1F)

**경사 하강법(Gradient Descent)**는 함수의 기울기를 이용하여 낮은 쪽으로 극값에 이를때까지 반복하는 최적화 알고리즘이다.

$i$는 현재 스텝이고, $x_i$최적화 할 변수의 현재 스텝, $\eta$는 learning rate, $f$는 함
$$
x_{i+1} = x_i-\eta \nabla f(x_i) \\
= x_i-\eta\frac{d f(x_i)}{d x_i}
$$


##### 2-4-3-2. Delta  rule

**델타 규칙**(delta rule)은 경사 하강법(Gradient Descent) 학습 방법으로, 단층 퍼셉트론에서 인공 뉴런들의 연결강도를 갱신하는데 쓰인다. 

| 변수            | 설명                                                     |
| --------------- | -------------------------------------------------------- |
| $t_k$           | 데이터 정답                                              |
| $y_k$           | 신경망 출력값                                            |
| $\Delta w_{ji}$ | j번 노드에서 i번 노드로 가는 가중치를 업데이트 할 변화량 |
| $\eta$          | 학습 상수 : learning rate                                |
| $g$             | 활성화 함수                                              |
| $x_k$           | 입력값                                                   |
| $h_j$           | $\sum_k x_k w_{jk}$                                      |


$$
E=\frac{1}{2}\sum_k(t_k-y_k)^2
\\
\Delta w_{ji}=\eta \frac{\partial E}{\partial w_{ji}}
\\
=\eta \frac{\partial (\frac{1}{2}(t_j-y_j)^2)}{\partial w_{ji}}
\\
=\eta \frac{\partial (\frac{1}{2}(t_j-y_j)^2)}{\partial y_j} \frac{\partial y_j}{\partial w_{ji}}
\\
= - \eta(t_j-y_j) \frac{\partial y_j}{\partial w_{ji}}
\\
= - \eta(t_j-y_j) \frac{\partial y_j}{\partial h_j} \frac{\partial h_j}{\partial w_{ji}}
\\
= - \eta(t_j-y_j)a'(h_j)\frac{\partial h_j}{\partial w_{ji}}
\\
= - \eta(t_j-y_j)a'(h_j)\frac{\partial (\sum_k x_k w_{jk}+ b)}{\partial w_{ji}}
\\
= - \eta(t_j-y_j)a'(h_j)\frac{\partial x_iw_{ji}}{\partial w_{ji}}
\\
= - \eta(t_j-y_j)a'(h_j)x_i
\\
\therefore \Delta w_{ji}= - \eta(t_j-y_j)a'(h_j)x_i
$$

위의 변화식을 이용하여 다음과 같이 가중치$w_{ij}$를 업데이트 할 수 있다.
$$
w_{ji}\leftarrow w_{ji}-\Delta w_{ji}
$$


##### 2-4-3-3. Back Propagation

l

## 3. Deep Neural Network

### 3-1. Deep Neural Network

### Convolution Neural Networks

### Recurrent Neural Networks

### Long Short-Term Memory models
