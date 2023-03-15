# c-bpnn

## 1.BP神经网络

+ 神经网络：由简单的神经元组成的广泛互联的网络，其具有适应性，可以模拟生物神经系统对真实世界所做出的交互反应。

- 多层前馈神经网络(multi-layer feedforward neural networks):基本结构如图3所示，此种结构的圣经网络，每层神经元与下一层神经元全连接，神经元之间不存在同层连接，也不存在跨层连接。其网络层次包含3层，其中输入层神经元用于接收外界输入，隐层与输出层神经元对信号进行加工，最终结果由输出层神经元输出。

<img src="./c-bpnn/4.png" style="zoom:50%;" />

<div style="text-align:center;">图1.1 多层前馈神经网络</div>

### 1.1BP神经网络结构

​		BP神经网络是目前为止最为成功的神经网络算法之一，其学习方式采用标准梯度下降的误差逆传播（error BackPropagation）的方式，下面介绍的基本BP神经网络为3层前馈神经网络。

<img src="./c-bpnn/2.png" style="zoom:50%;" />

<div style="text-align:center;">图1.2.1 三层BP神经网络模型</div>

​		对于BP神经网络，我们需要使用训练数据集对其进行参数训练，然后使用测试机检验训练结果，如果训练效果达标，则可使用训练出的数据应用于实际使用场景。

​		对于图1.2.1中的神经网络模型，我们做如下定义：

+ 给定训练集D={(x1,y1),(x2,y2),...,(xm,ym)},XiϵR^d^,YiϵR^l^，即输入数据维度为d，输出数据维度为l。
+ 图1.2.1中，假设神经网络是输入节点为d个，输出节点为l个，隐层有q个的多层前向反馈神经网络，输出层第j个神经元阈值为θj，第h个隐层神经元阈值为γh。输入层第i个节点与隐层第h个节点之间的权重为V~ih~，隐层第h个节点与输出层第j个节点的权重为W~hj~。

**根据以上假设可以有如下公式：**

1. 激活函数为f(x)=sigmoid(x)

<img src="./c-bpnn/3.png" style="zoom:50%;" />

<div style="text-align:center;">图1.2.2 sigmoid函数</div>

2. 隐层第h个神经元接收到的输入为α<sub>h</sub>=∑<sub>i=1</sub> V~ih~X~i~

3. 隐层第h个神经元的输出为b~h~=f(α~h~−γ~h~)

4. 输出层第j个神经元接收到的输入为β~j~=∑~h=1~w~hj~b~h~

5. 输出层第j个神经元的输出为y~i~=f(β~j~-θ~j~)

​		由以上5个公式可知，神经网络之中只要*(d+l+1)q+l*个参数确定，则就可以由输入计算出输出，这些参数分别为

1. 输入层到隐层权重dq个，隐层到输出层权重ql个
2. 隐层神经元阈值q个，输出层神经元阈值l个

### 1.2BP神经网络的训练

​		神经网络的*初始参数为[0,1]内随机数*，假设某次训练过程中神经网络的输入的某个训练数据为(x,y),经过神经网络的输出为yj=f(βj−θj)

对于训练数据集中的单个数据其误差 6. <img src="./c-bpnn/5.png" style="zoom:50%;" />

采用梯度下降法根据误差对神经网络中的参数进行反馈学习，神经网络中参数更新的公式为**p←p+Δp**

以误差的负梯度方向对参数进行更新，η为学习率，有如下公式：

<img src="./c-bpnn/6.png" style="zoom:50%;float:left;" />

按照以上推导方法有

<img src="./c-bpnn/7.png" style="zoom:50%;float:left;" />

**注意：**在公式13.14.15.16中η控制着每一轮迭代中的更新步长，若太大，则容易振荡，太小则学习过程收敛很慢，为了细微调节，13.14中的学习率可以和15.16中的不一样。

<img src="./c-bpnn/8.png" style="zoom:50%;float:left;" />

### 1.3 **BP神经网络的训练过程**

+ 在(0,1)内随机初始化神经网络中的连接权重和阈值
+ 重复
+ for 遍历训练集中的每一个样本
+ 根据当前参数按照公式1.2.3.4.5计算样本的bh与yj.
+ 根据公式12.17计算gj与eh.
+ 根据公式13.14.15.16更新whj,vih,θj,γh.
+ end for
+ until (迭代若干次或者累积误差E小于特定值)得到BP神经网络的参数

## 2. 代码实现

### 2.1 数据集

​		该数据集使用的是歌词文本特征提取，用以识别音乐情感分类。

### 2.2 代码实现

**训练模块**

```c

/**
 * To fetch test_set.
 * The length of in is IN_N, the length of out is OUT_N.
 */
typedef bool (*test_set_get_t)(double *in, double *out);

/**
 * Reset test_set fetch process.
 */
typedef bool (*test_set_init_t)(void);

/**
 * Init bpnn module.
 */
void bpnn_init(void);

/**
 * Train bpnn module and produce parameter file.
 * @param f_get To get test data in stream.
 */
void bpnn_train(test_set_get_t f_get, test_set_init_t f_init);


```

**拟合模块**

```c

typedef struct T *T;

/**
 * Init bpnn module.
 * @return
 */
T bpnn_fit_new(void);

/**
 * Using bpnn fit in to out.
 * @param bpnn
 * @param in
 * @param out
 */
void bpnn_fit(T bpnn, double *in, double *out);

/**
 * Uninit bpnn.
 * @param bpnn
 */
void bpnn_fit_free(T *bpnn);

  
```

+ 配置文件

```c


#define IN_N                            600           /* INPUT NODE */
#define OUT_N                           4           /* OUTPUT_NODE */
#define HIDDEN_N                        200          /* HIDDEN_NODE */
#define LOOP_N                          5000        /* LOOP NUMBER */
#define E_MIN                           0.000001    /* Cumulative error */
#define LEARN_RATE1                     0.3
#define LEARN_RATE2                     0.4

#define ACTIVATION_FUNC(x)              (1/(1+exp(-(x)))) /* Sigmoid */

#define TEST_IN_PATH                    "../dataset/test_file.txt"
#define TEST_OUT_PATH                   "../dataset/test_label_file.txt"
#define IN_PATH                         "../dataset/train_file.txt"
#define OUT_PATH                        "../dataset/train_label_file.txt"
#define SAVE_PARAM_PATH                 "../dataset/bpnn_param.txt"

```

+ 初始化权重和阀值

```c
void bpnn_init(void) {
  // 随机种子
    srand((unsigned) time(NULL));
    for (size_t i = 0; i < D; i++)
        for (size_t h = 0; h < Q; h++)
            v[i][h] = RAND;
    for (size_t h = 0; h < Q; h++)
        for (size_t j = 0; j < L; j++)
            w[h][j] = RAND;
    for (size_t h = 0; h < Q; h++)
        r[h] = RAND;
    for (size_t j = 0; j < L; j++)
        o[j] = RAND;
}
```

+ 前向传播计算输出

```c
/* Compute b[h] */
    for (size_t h = 0; h < Q; h++) {
        double alpha_h = 0;
        for (size_t i = 0; i < D; i++)
            alpha_h += v[i][h] * x[i];
        b[h] = ACTIVATION_FUNC(alpha_h - r[h]);
    }

    /* Compute yc[j] */
    for (size_t j = 0; j < L; j++) {
        double beta_j = 0;
        for (size_t h = 0; h < Q; h++)
            beta_j += w[h][j] * b[h];
        yc[j] = ACTIVATION_FUNC(beta_j - o[j]);
    }
```

+ 反向传播，更新权重值

```c
/* Compute g[j] and E */
    for (size_t j = 0; j < L; j++) {
        g[j] = yc[j] * (1 - yc[j]) * (y[j] - yc[j]);
        Ek += (yc[j] - y[j]) * (yc[j] - y[j]);
    }

    Ek = 0.5 * Ek;

    /* Compute e[h] */
    for (size_t h = 0; h < Q; h++) {
        double temp = 0;
        for (size_t j = 0; j < L; j++)
            temp += w[h][j] * g[j];
        e[h] = b[h] * (1 - b[h]) * temp;
    }

    /* Update v[i][h], w[h][j], r[h], o[j] */
    for (size_t i = 0; i < D; i++)
        for (size_t h = 0; h < Q; h++)
            v[i][h] += U2 * e[h] * x[i];
    for (size_t h = 0; h < Q; h++)
        for (size_t j = 0; j < L; j++)
            w[h][j] += U1 * g[j] * b[h];
    for (size_t h = 0; h < Q; h++)
        r[h] += ((-U2) * e[h]);
    for (size_t j = 0; j < L; j++)
        o[j] += ((-U1) * g[j]);

```

### 2.3 实验结果

**手写数字识别



<img src="./c-bpnn/9.png" style="zoom:50%;" />

<div style="text-align:center;">图2.3.1 BP训练过程</div>

​		积累误差保持在2.009756（这个原因第一是数据量少，第二是数据分布不均衡）

