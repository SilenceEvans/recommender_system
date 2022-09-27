PNN的构建参考了大牛吴忠强的[实现过程](https://github.com/zhongqiangwu960812/AI-RecommenderSystem/tree/master/Rank/PNN)，但是在PNN具体实现的时候，Product模块参数矩阵没有使用原代码提出的初始化三个矩阵的方法，而是采用了三个全连接层，因为在理解了PNN关于embedding部分的三个操作之后，我觉得最终无非是对操作后的矩阵进行了一个和参数矩阵相乘的操作，也就是全连接的操作，如果初始化三个参数矩阵之后，再对embedding矩阵和参数矩阵同时都进行transpose或者permute的操作极容易让人头大，既然知道embedding矩阵各种变化之后的形状，那也就指导了它们应该进行什么样的线性操作，还不如直接构建三个全连接层，暂时是这么考虑的，不知道这样做有没有其他问题。

> embedding部分之后的三个操作主要是：
>
> - concatenation
> - Inner product
> - outer product：外积之前接触的少，导公式的时候导的有点麻烦

​																																					——2022年09月11日