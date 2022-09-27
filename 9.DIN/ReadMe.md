## 数据预处理

数据预处理模块采用的数据来自亚马逊产品数据集里面的Electronics子集，reviews_Electronics.json和meta_Electronics.json这俩文件数据量太大了，所以不考虑上传到github，这俩文件下载地址在：

- http://deepyeti.ucsd.edu/jianmo/amazon/index.html

## 模型的定义

搭建模型的过程中参考了吴忠强大佬的[TensorFlow代码](https://github.com/zhongqiangwu960812/AI-RecommenderSystem/tree/master/Rank/DIN)，将里面模型搭建的过程替换成了torch的代码，(大佬在22年2月的博客中写道自己会将keras的代码之后换成torch的，但github最新的代码依然是keras的...)因为之前对keras的代码不是很熟悉，前几天像WideDeep等一两个小时就能搭建好的模型，这次搭建DIN整整用了四天，但整个过程也是获益颇多，尤其是在dataset定义过程中对用户历史序列padding的处理，以及后面为了避免paddding的值在softmax过程中也参与计算进行mask等的操作，让自己对之前NLP里面的一些编码知识又有了新的认识，总之来说这几天花的值。

本次的代码因为自己目前所学的原因肯定还是有不太完善的地方，但将此套推荐系统深度学习模型全部用Pytorch实现也是我的一个初步目标，之后有时间还会继续修改。

