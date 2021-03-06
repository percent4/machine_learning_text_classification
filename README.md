本项目使用机器学习模型进行文本分类。

### 数据集

#### Sougou-Mini数据集

使用sougou_mini数据集进行训练与测试，5个分类，每个分类1000条数据。

类别如下：

体育, 健康, 汽车, 军事, 教育

数据集划分如下：

训练集: 800 \* 5
验证集: 100 \* 5  
测试集: 100 \* 5

### 模型

利用ALBERT对文本进行特征提取，每句话转化为312维向量。再使用LR, NB, SVM算法进行分类。

### 模型评估

```
Logistic Regression Model
混淆矩阵 [[88  0  7  2  2]
 [ 0 91  2  4  2]
 [ 0  0 96  2  1]
 [ 2  6  1 89  1]
 [ 0  1  3  2 93]]
正确率： 0.9232323232323232
              precision    recall  f1-score   support

          体育     0.9778    0.8889    0.9312        99
          健康     0.9286    0.9192    0.9239        99
          军事     0.8807    0.9697    0.9231        99
          教育     0.8990    0.8990    0.8990        99
          汽车     0.9394    0.9394    0.9394        99

    accuracy                         0.9232       495
   macro avg     0.9251    0.9232    0.9233       495
weighted avg     0.9251    0.9232    0.9233       495


Naive Bayes Model
混淆矩阵 [[81  0  5 10  3]
 [ 0 81  1 11  6]
 [ 0  0 95  1  3]
 [ 1  9  0 85  4]
 [ 3  4  4  7 81]]
正确率： 0.8545454545454545
              precision    recall  f1-score   support

          体育     0.9529    0.8182    0.8804        99
          健康     0.8617    0.8182    0.8394        99
          军事     0.9048    0.9596    0.9314        99
          教育     0.7456    0.8586    0.7981        99
          汽车     0.8351    0.8182    0.8265        99

    accuracy                         0.8545       495
   macro avg     0.8600    0.8545    0.8552       495
weighted avg     0.8600    0.8545    0.8552       495


SVM Model
混淆矩阵 [[95  0  0  1  3]
 [ 0 93  0  5  1]
 [ 0  0 98  0  1]
 [ 1  5  0 93  0]
 [ 1  1  1  3 93]]
正确率： 0.9535353535353536
              precision    recall  f1-score   support

          体育     0.9794    0.9596    0.9694        99
          健康     0.9394    0.9394    0.9394        99
          军事     0.9899    0.9899    0.9899        99
          教育     0.9118    0.9394    0.9254        99
          汽车     0.9490    0.9394    0.9442        99

    accuracy                         0.9535       495
   macro avg     0.9539    0.9535    0.9536       495
weighted avg     0.9539    0.9535    0.9536       495
```

### 模型预测

```
预测类别: 军事, 句子: 9月底，美国媒体盛传特朗普政府为制造大选翻盘的“十月惊奇”，可能会派出MQ-9无人机部队对南海的中国岛礁实施打击。与此同时，网络上流出的一组照片清晰地显示，美军MQ-9无人机部队上的臂章秀出了中国地图。
预测类别: 汽车, 句子: 在三年的时间里，我们的参赛车型数量由最初的7台跃升至今年的20台，代表车型也由曾经的腾势500、比亚迪元EV(参数|图片)、帝豪EV变成了如今的特斯拉Model 3(参数|图片)、极星2、蔚来EC6。车型数量以及综合素质的大踏步式提升也让作为新能源车媒体从业者的我倍感欣慰。
预测类别: 健康, 句子: 天气转凉，不少人出现了口唇起皮、眼睛干涩、口渴难忍、皮肤干燥等不适症状。有一类干燥症状不仅持续性存在，且“干燥”程度逐渐加重，甚至出现发热、牙齿脱落、关节痛、皮疹等症状，此时需要警惕患有干燥综合征的可能。今天北京中医药大学东方医院风湿科韦尼医生来详细给大家谈谈恼人的干燥综合征。
预测类别: 教育, 句子: 根据《中华人民共和国中外合作办学条例》及其实施办法，教育部对2020年上半年各地上报的中外合作办学项目进行了评审。依照专家评议结果，经研究，决定批准32个本科以上中外合作办学项目。其中本科层次合作办学教育项目共计27个，硕士层次合作办学项目共计5个。
预测类别: 体育, 句子: 北京时间11月4日，来自《纽约时报》名记马克-斯坦恩的报道，有消息人士透露，火箭当家球星詹姆斯-哈登对于球队新的主帅选择并不满意。据悉，哈登更希望泰伦-卢或者约翰-卢卡斯二世成为火箭新主帅，然而，火箭最终却选了独行侠助教斯蒂芬-塞拉斯。
```

### 补充

在THUCNews数据集上的模型结果：

```
Logistic Regression Model
混淆矩阵 [[982   3   0   1   2   0   2   7   2   1]
 [  0 961   4   0   3   7   3  14   6   2]
 [  1   7 511 281  20  67  26  15  42  30]
 [  1   4  21 846  19  10  43   5   5  46]
 [  7   4  11  21 845   2  23  47  36   4]
 [  1  11  22   1   7 946   1   4   6   1]
 [  0   7   1  31  16   0 916   3  13  13]
 [  0   4   4   4  12  12   1 953   9   1]
 [  0   2  10   3   1  15   0  12 953   4]
 [  1   0   3  26   3   0   6   0   0 961]]
正确率： 0.8874
              precision    recall  f1-score   support

          体育     0.9889    0.9820    0.9854      1000
          娱乐     0.9581    0.9610    0.9596      1000
          家居     0.8705    0.5110    0.6440      1000
          房产     0.6969    0.8460    0.7642      1000
          教育     0.9106    0.8450    0.8766      1000
          时尚     0.8933    0.9460    0.9189      1000
          时政     0.8972    0.9160    0.9065      1000
          游戏     0.8991    0.9530    0.9252      1000
          科技     0.8890    0.9530    0.9199      1000
          财经     0.9040    0.9610    0.9317      1000

    accuracy                         0.8874     10000
   macro avg     0.8908    0.8874    0.8832     10000
weighted avg     0.8908    0.8874    0.8832     10000


Naive Bayes Model
混淆矩阵 [[917   8  25   0   5   0  16  25   3   1]
 [  1 914  25   2   3   6  18  24   4   3]
 [  0  20  88 419 115  89  93  66  98  12]
 [  0  13  27 692  37   5  93  13   6 114]
 [  4  18  54  51 754   1  37  42  22  17]
 [  0  22 122   1   6 809   2  25  13   0]
 [  0  29   6  51  28   0 852   3  11  20]
 [  6   6  11   3  11  23   6 925   8   1]
 [  0   2 132   1   2  23   2  30 807   1]
 [  0   0   8  44   3   0  47   0   1 897]]
正确率： 0.7655
              precision    recall  f1-score   support

          体育     0.9881    0.9170    0.9512      1000
          娱乐     0.8857    0.9140    0.8996      1000
          家居     0.1767    0.0880    0.1175      1000
          房产     0.5475    0.6920    0.6113      1000
          教育     0.7822    0.7540    0.7678      1000
          时尚     0.8462    0.8090    0.8272      1000
          时政     0.7307    0.8520    0.7867      1000
          游戏     0.8023    0.9250    0.8593      1000
          科技     0.8294    0.8070    0.8180      1000
          财经     0.8415    0.8970    0.8683      1000

    accuracy                         0.7655     10000
   macro avg     0.7430    0.7655    0.7507     10000
weighted avg     0.7430    0.7655    0.7507     10000

 
SVM Model
混淆矩阵 [[983   4   0   1   4   1   2   3   2   0]
 [  0 960   3   0   5   8   4  10   6   4]
 [  0   7 480 342  21  68  26  11  20  25]
 [  1   4  20 831  23  15  47   5   3  51]
 [  5   5   4  21 869   4  15  46  25   6]
 [  1  10  22   1  10 942   1   6   6   1]
 [  0   9   0  29  20   0 922   1   7  12]
 [  0   5   2   4  15   9   2 957   6   0]
 [  0   3   7   2   1   9   0  10 964   4]
 [  0   0   3  17   3   0   7   0   0 970]]
正确率： 0.8878
              precision    recall  f1-score   support

          体育     0.9929    0.9830    0.9879      1000
          娱乐     0.9533    0.9600    0.9567      1000
          家居     0.8872    0.4800    0.6230      1000
          房产     0.6659    0.8310    0.7393      1000
          教育     0.8950    0.8690    0.8818      1000
          时尚     0.8920    0.9420    0.9163      1000
          时政     0.8986    0.9220    0.9102      1000
          游戏     0.9123    0.9570    0.9341      1000
          科技     0.9278    0.9640    0.9456      1000
          财经     0.9040    0.9700    0.9358      1000

    accuracy                         0.8878     10000
   macro avg     0.8929    0.8878    0.8831     10000
weighted avg     0.8929    0.8878    0.8831     10000
```