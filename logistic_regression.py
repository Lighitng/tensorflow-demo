import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import read_xlsx_data as rd
from sklearn.preprocessing import  StandardScaler # 标准化
from sklearn.preprocessing import scale # 按行或按列标准化
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn import  datasets


# 将表格中的数据导入
datas = rd.read_excel_sync()
data = datas[0]['data'].values
target_data = np.asarray([1.0 if x != 0 else 1.0 for x in data])

'''
# 训练组数的定义
BATCH_SIZE = 60
'''

# 将表格中每个水稻品类的六列属性外加时间作为七组特征值的输入
# 每一个特征值都是有BATCH_SIZE行，一列
'''
x1_data = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="x1")
x2_data = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="x2")
x3_data = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="x3")
x4_data = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="x4")
x5_data = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="x5")
x6_data = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="x6")
x7_data = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="x7")
'''

# 定义神经网络的参数
learning_rate = 0.009  # 学习率
training_step = 1000  # 训练迭代次数
testing_step = 500  # 测试迭代次数
display_step = 100  # 每多少次迭代显示一次损失

# 定义输入和输出
x = tf.placeholder(shape=(None, 7), dtype=tf.float32, name="X_train")
y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="Y_train")

# 定义模型参数
w = tf.Variable(tf.random_normal([7, 1], stddev=1.0, seed=1))
b = tf.Variable(tf.random_normal([1], stddev=1.0, seed=1))

# 定义神经网络前向传播过程，以下两种模型也是可以的，但是对于二元分类来讲，最常用的就是sigmoid函数模型
# Model = tf.nn.tanh(tf.matmul(x,w) + b)
# Model = tf.nn.relu(tf.matmul(x,w) + b)
Model = tf.nn.sigmoid(tf.matmul(x, w) + b)

"""
对模型进行优化，将Model的值加0.5之后进行取整，
方便测试准确率(若Model>0.5则优化后会取整为1，反之会取整为0)
"""
model = Model + 0.5
model = tf.cast(model, tf.int32)
y_ = tf.cast(y, tf.int32)

# Dropout操作：用于防止模型过拟合
keep_prob = tf.placeholder(tf.float32)
Model_drop = tf.nn.dropout(Model, keep_prob)

# 损失函数：交叉熵
cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(Model, 1e-10, 1.0)) + (1-y) * tf.log(tf.clip_by_value(1-Model, 1e-10, 1.0)))

"""
优化函数
即反向传播过程
主要测试了Adam算法和梯度下降算法，Adam的效果较好
"""
# 优化器：使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# 优化器：梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# 进行数据预处理
# 训练集
X_train = data[1:80]
Y_train = target_data[1:80]

# 测试集
X_test = data[90:100]
Y_test = target_data[90:100]

b = MinMaxScaler()
X_test_cen = b.fit_transform(X_test)

a = MinMaxScaler()
X_train_cen = a.fit_transform(X_train)

# 计算精准度
correct_prediction = tf.equal(model, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求所有correct_prediction的均值

# 创建会话运行Tensorflow程序
with tf.Session() as sess:
    init_all_op = tf.global_variables_initializer()
    sess.run(init_all_op)
    writer = tf.summary.FileWriter("./log/", tf.get_default_graph())
    for i in range(training_step):
        sess.run(optimizer, feed_dict={x: X_train_cen, y: Y_train, keep_prob: 0.5})  # 训练模型语句（采用矩阵运算将训练时间减少至十几秒）
        # 每迭代1000次输出一次日志信息
        if i % display_step == 0:
            # 输出交叉熵之和
            total_cross_entropy_train = sess.run(cross_entropy,feed_dict={x: X_train_cen, y: Y_train})
            print("After &d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy_train))
            # 输出准确度
            # 每10轮迭代计算一次准确度
            accuracy_rate = sess.run(accuracy, feed_dict={x: X_train_cen, y: Y_train, keep_prob:1.0})
            print('第' + str(i) + '轮， Training 的准确度为：' + str(accuracy_rate))

# 通过选取样本训练神经网络并更新参数
    for i in range(testing_step):
        total_cross_entropy_test = sess.run(cross_entropy, feed_dict={x: X_test_cen, y: Y_test})
        print("After &d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy_test))
        accuracy_rate1 = sess.run(accuracy, feed_dict={x: X_test_cen, y: Y_test, keep_prob: 1.0})
        print('第' + str(i) + '轮,Testing的准确度为：' + str(accuracy_rate1))





















