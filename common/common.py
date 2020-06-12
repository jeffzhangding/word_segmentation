__author__ = 'jeff'


def tow_tag(words):
    tag_list = []
    word_list = []
    for word in words:
        if word == ' ' or word == '':
            continue
        # if len(word) == 1:
        #     tag_list.append('S')
        # elif len(word) > 1:
        tag_list.append('B' + 'I' * (len(word) - 1))
        word_list.append(word)
    return ''.join(word_list), ''.join(tag_list)


def four_tag(words):
    """"""
    tag_list = []
    word_list = []
    for word in words:
        if word == ' ' or word == '':
            continue
        if len(word) == 1:
            tag_list.append('S')
        elif len(word) == 2:
            tag_list.append('BE')
        else:
            m_tag = ''.join(['M']*(len(word)-2))
            tag_list.append('B%sE' % m_tag)
        word_list.append(word)
    return ''.join(word_list), ''.join(tag_list)
