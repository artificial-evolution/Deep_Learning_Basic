# Deep Learning Basics

## Contents

1. Machine Learning
   1. Concept of Machine Learning
   2. Types of learning algorithms
      1. Supervised Learning
      2. Unsupervised Learning
      3. Reinforcement Learning
   3. Machine Learning models
2. Deep Learning and Artificial Neural Network
   1. Deep Learning
   2. Artificial Neural Network
      1. Concept of ANN and DNN
      2. Structure of ANN
   3. Perceptron
      1. Concept of Perceptron
      2. Structure of Perceptron
         1. Basic Structure
         2. Activation functions
         3. Cost functions / Loss functions
         4. Learning Algorithms
      3. XOR Problem
   4. Multi Layer Perceptron
      1. Concept of MLP
      2. Structure of MLP
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

지도 학습이란, 입력 레이블과 레이블에 대한 정답으로 이루어진 학습 데이터를 통하여 해당하는 입력에 대해 정답이 나올 수 있도록 학습한 후, 결과적으로 어떠한 입력에 대해 원하는 정답을 예측할 수 있도록 하는 학습 방법이다.

각 학습 데이터에는 명확한 해답이 사용되며, 정답 레이블이 이산 값이면 Classification(분류), 연속 값이면 Regression(회귀)라고 불린다.

ex) softmax regression

지도학습을 진행할 때는 주로 CNN과 RNN과 같은 ANN 모델이 사용된다.

#### 1-2-2. Unsupervised Learning

비 지도 학습이란, 입력 데이터만 주어지고, 입력 데이터들의 숨겨진 특성(hidden feature)이나 구조를 발견하는 방법이다.

입력값에 대한 명확한 해답이 없다는 부분에서 지도학습과 차이를 보인다.

Clustering 등의 알고리즘이 비 지도 학습에 속하며, 주로 Auto Encoder라는 ANN모델이 사용된다.

#### 1-2-3. Reinforcement Learning

강화 학습이란, 행동 심리학에서 영향을 받았으며, 현재 상태에서 할 수 있는 행동들이 주어지고, 각 상황에 따른 보상이 주어지는 환경에서 에이전트가 정의되어있을 때, 에이전트가 해당 환경에서 보상을 최대화 하거나, 보상을 최대화 하는 행동 순서를 선택하는 방법이다.

강화학습 모델로 DQN을 많이 사용한다.

### 1-3. Machine Learning models



#### 1-3-1. Artificial Neural Network



## 2. Deep Learning and Artificial Neural Network

### 2-1. Deep Learning

### 2-2. Artificial Neural Network

#### 2-2-1. Concept of ANN

#### 2-2-2. Structure of ANN

### 2-3. Perceptron

#### 2-3-1. Concept of Perceptron

#### 2-3-2. Structure of Perceptron

##### 2-3-2-1. Basic Structure

##### 2-3-2-2. Activation functions

##### 2-3-2-3. Cost functions / Loss functions

##### 2-3-2-4. Learning Algorithms

#### 2-3-3. XOR Problem

### 2-4. Multi Layer Perceptron

#### 2-4-1. Concept of MLP

#### 2-4-2. Structure of MLP

##### 2-4-2-1. Basic Structure

##### 2-4-2-2. Learning Algorithm : Back Propagation

## 3. Deep Neural Network

### 3-1. Deep Neural Network

### Convolution Neural Networks

### Recurrent Neural Networks

### Long Short-Term Memory models