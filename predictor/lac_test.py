__author__ = 'jeff'

from LAC import LAC

# 装载分词模型
# lac = LAC(mode='seg')
#
# lac.load_customization('custom.txt')
#
# # 单个样本输入，输入为Unicode编码的字符串
# text = u"LAC是个优秀的分词工具"
# seg_result = lac.run(text)
#
# # 批量样本输入, 输入为多个句子组成的list，速率会更快
# texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
# seg_result = lac.run(texts)
#
# print(seg_result)

def test():
    from LAC import LAC
    lac = LAC()

    custom_result = lac.run(u"春天的花开秋天的风以及冬天的落阳")
    print('装载前： === %s' % str(custom_result))

    # 装载干预词典
    lac.load_customization('custom.txt')
    custom_result = lac.run(u"春天的花开秋天的风以及冬天的落阳")
    print('装载后： === %s' % str(custom_result))


if __name__ == '__main__':
    test()


