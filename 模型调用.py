import os
import time


from datetime import timedelta

import tensorflow as tf
import numpy as np

from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN

from data.cnews_loader import read_vocab, read_category, batch_iter, process_single_URL, build_vocab


# base_dir = 'data/cnews'
# train_dir = os.path.join(base_dir, 'cnews.train.txt')
# test_dir = os.path.join(base_dir, 'cnews.test.txt')
# val_dir = os.path.join(base_dir, 'cnews.val.txt')
# vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

base_dir = 'data/CSIC2010'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')


save_dir = 'checkpoints/textrnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

print('Configuring CNN model...')
config = TCNNConfig()
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(train_dir, vocab_dir, config.vocab_size)
categories, cat_to_id = read_category()  # 获得名称到id的映射
words, word_to_id = read_vocab(vocab_dir)  # 获得词汇到id的映射

config.vocab_size = len(words)
config.num_classes = len(categories)
model = TextCNN(config)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def anomaly_test(URL):
    print("Loading test data...")
    start_time = time.time()
    x_test = process_single_URL(URL, word_to_id, cat_to_id, config.seq_length) #6731,500

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    #loss_test, acc_test = evaluate(session, x_test, y_test)#    6731,500     6731,2
    #msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    #print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)   #6731
    num_batch = int((data_len - 1) / batch_size) + 1

    #y_test_cls = np.argmax(y_test, 1)  #[6731]
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # [6731]保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    print(y_pred_cls)

    # 评估
    #print("Precision, Recall and F1-Score...")

    #print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    #print("AUC...")
    #print(sklearn.metrics.roc_auc_score(y_test_cls, y_pred_cls))


    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    anomaly_test("123")