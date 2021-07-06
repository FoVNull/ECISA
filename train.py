import kashgari
import jieba
import re
from lxml import etree
from kashgari.tasks.classification import BiLSTM_Model
from kashgari.embeddings import BertEmbedding
import tensorflow as tf
from sklearn.model_selection import train_test_split
from RCNN_Att import RCNN_Att_Model

import logging
logging.basicConfig(level='DEBUG')

stop_words = set()
with open('./stop_words.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop_words.add(line.strip())


def cut_text(_line):
    _line = "".join(re.split("[//]*[\\\\]*@[\s\S]+:", _line))
    # _line = "".join(re.findall("[\u4e00-\u9fa5]", _line))
    vocab_list = [word for word in jieba.lcut(_line, cut_all=False, use_paddle=True) if
                  word not in stop_words]
    return vocab_list


def read_data(path):
    x = []
    y = []
    train_xml = etree.parse(path)
    root = train_xml.getroot()
    for doc in root:
        for sent in doc:
            label = sent.get('label')
            if not label:
                continue
            x.append(cut_text(sent.text))
            y.append(label)
    return x, y


def read_test():
    x = []
    _num = []
    train_xml = etree.parse('./data/SMP2019_ECISA_Test.xml')
    root = train_xml.getroot()
    for doc in root:
        did = doc.get('ID')
        for sent in doc:
            sid = sent.get('ID')
            _num.append("{}-{}".format(did, sid))
            x.append(cut_text(sent.text))
    return x, _num


def gen_interface(pred, _num):
    with open('result.txt', 'w', encoding='utf-8') as f:
        for i in range(len(pred)):
            f.write(_num[i]+"\t"+str(pred[i])+"\n")


if __name__ == '__main__':

    with tf.device('/gpu:3'):
        train_x, train_y = read_data('./data/SMP2019_ECISA_Train.xml')
        dev_x, dev_y = read_data('./data/SMP2019_ECISA_Dev.xml')
        dev_x, test_x, dev_y, test_y = train_test_split(dev_x, dev_y, test_size=0.5)

        interface_x, num = read_test()
        embedding = BertEmbedding("chinese_L-12_H-768_A-12")
        model = RCNN_Att_Model(embedding)
        model.fit(train_x, train_y, dev_x, dev_y, epochs=24, batch_size=32)

        # Evaluate the model
        model.evaluate(test_x, test_y)

        pred_y = model.predict(interface_x)
        gen_interface(pred_y, num)