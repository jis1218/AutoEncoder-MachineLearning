### Auto Encoder

##### Autoencoder는 unsupervised learning 중 하나로 자기 자신을 목표로 학습하면서 핵심을 찾아낸다.

##### loss function(cost function)은 다음과 같이 정의한다. 재구성한 이미지와 원래의 이미지가 얼마나 흡사한지를 측정해야 한다. 원래의 784차원 입력과 재구성한 784차원의 출력 사이의 거리를 계산함으로써 측정 가능하다.

##### batch regularization - 신경망에서 모든 가중치 w에 대해 1/2*lamda*w^2를 오차 함수에 추가 - 큰 W값은 견제하고 고르게 분포되어 있는 W값을 도와준다. 과적합을 막기 위함

##### AdamOptimizer의 파라미터는 어떤 값들인가?
##### beta1: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
##### beta2: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
##### epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.