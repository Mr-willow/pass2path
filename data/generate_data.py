import tensorflow as tf
import numpy as np
import json
from data.word2keyboard import switch2keyboard_list
from data.generate_path import find_med_backtrace

max_length = 20
min_length = 4
max_path_length = 5
char_list = [' ', '\n', '\x0b', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '`', '-', '=', '[', ']', '\\', ';',
             "'", ',', '.', '/', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\x0c', '\t']

# 96 ascii char and space \t \n
# \t and \n are used as the start and end of a password respectively
# space is used as the padding
# '\x0c' is caps, '\x0b' is shift
def get_char_vs_num():
    char2num = {' ': 0, '\n': 1, '\x0b': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11,
                '0': 12, '`': 13, '-': 14, '=': 15, '[': 16, ']': 17, '\\': 18, ';': 19, '\'': 20, ',': 21, '.': 22,
                '/': 23, 'a':24, 'b':25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33,
                'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44,
                'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '\x0c': 50, '\t': 51}
    num2char = {0: ' ', 1: '\n', 2: '\x0b', 3: '1', 4: '2', 5: '3', 6: '4', 7: '5', 8: '6', 9: '7', 10: '8', 11: '9',
                12: '0', 13: '`', 14: '-', 15: '=', 16: '[', 17: ']', 18: '\\', 19: ';', 20: "'", 21: ',', 22: '.',
                23: '/', 24: 'a', 25: 'b', 26: 'c', 27: 'd', 28: 'e', 29: 'f', 30: 'g', 31: 'h', 32: 'i', 33: 'j',
                34: 'k', 35: 'l', 36: 'm', 37: 'n', 38: 'o', 39: 'p', 40: 'q', 41: 'r', 42: 's', 43: 't', 44: 'u',
                45: 'v', 46: 'w', 47: 'x', 48: 'y', 49: 'z', 50: '\x0c', 51: '\t'}

    return char2num, num2char


def get_path_vs_id():
    path2id = {}
    id2path = {}
    id = 0
    for location in range(max_length):
        path = ('d', None, location)
        path2id[path] = id
        id2path[id] = path
        id += 1
        for edit in ['i', 's']:
            for element in char_list:
                path = (edit, element, location)
                path2id[path] = id
                id2path[id] = path
                id += 1

    return path2id, id2path


char2num, num2char = get_char_vs_num()
path2id, id2path = get_path_vs_id()


def generate_dataset(filename):

    train_dataset = []
    target_dataset = []

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            data = line.split('\t')
            # 处理数据时删除长度大于max_length和长度小于min_length的password
            if (not min_length <= len(data[1]) <= max_length) or (not min_length <= len(data[2]) <= max_length):
                line = f.readline()
                continue
            train_data = '\t' + data[1] + '\n'
            train_data = switch2keyboard_list(train_data)  # switch to keyboard
            train_vector = []
            for i in range(max_length):
                try:
                    train_vector.append(char2num[train_data[i]])
                except:
                    train_vector.append(char2num[' '])
            train_dataset.append(train_vector)

            target_data = '\t' + data[2] + '\n'
            target_data = switch2keyboard_list(target_data)  # switch to keyboard
            target_vector = []
            for i in range(max_length):
                try:
                    target_vector.append(char2num[target_data[i]])
                except:
                    target_vector.append(char2num[' '])
            target_dataset.append(target_vector)

            line = f.readline()

    train_dataset = np.array(train_dataset)
    target_dataset = np.array(target_dataset)
    train_dataset = tf.convert_to_tensor(train_dataset)
    target_dataset = tf.convert_to_tensor(target_dataset)

    return train_dataset, target_dataset


def generate_bucket_list(filename):

    train_dataset = []
    target_dataset = []
    train_bucket_dict = {}
    target_bucket_dict = {}

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            data = line.split('\t')
            # 处理数据时删除长度大于max_length和长度小于min_length的password
            if (not min_length <= len(data[1]) <= max_length) or (not min_length <= len(data[2]) <= max_length):
                line = f.readline()
                continue
            train_data = '\t' + data[1] + '\n'
            target_data = '\t' + data[2] + '\n'
            train_data = switch2keyboard_list(train_data)  # switch to keyboard
            target_data = switch2keyboard_list(target_data)  # switch to keyboard
            length = len(train_data)  # the length of pass1
            ed, path = find_med_backtrace(train_data, target_data)
            if not (1 <= len(path) <= max_path_length):  # filter the meaningless data
                line = f.readline()
                continue
            train_vector = []
            target_vector = []

            # char to int
            for i in range(length):
                train_vector.append(char2num[train_data[i]])
            # padding the target vector to same length
            for i in range(max_path_length):
                try:
                    target_vector.append(path2id[path[i]])
                except:
                    target_vector.append(path2id[('i', ' ', 1)])
            target_vector.insert(0, path2id[('i', ' ', 0)])
            # for i in range(len(target_data)):
            #     train_vector.append(char2num[target_data[i]])

            if length not in train_bucket_dict:
                train_bucket_dict[length] = [train_vector]
                target_bucket_dict[length] = [target_vector]
            else:
                train_bucket_dict[length].append(train_vector)
                target_bucket_dict[length].append(target_vector)

            line = f.readline()

    # trans list to tensor
    for l in train_bucket_dict:
        train_bucket_dict[l] = tf.convert_to_tensor(np.array(train_bucket_dict[l]))
        target_bucket_dict[l] = tf.convert_to_tensor(np.array(target_bucket_dict[l]))

    return train_bucket_dict, target_bucket_dict


if __name__ == '__main__':
    train_dict, target_dict = generate_bucket_list('csdn_dodonew_reuse_uniq.txt')
    print(target_dict)
    # d1, d2 = get_path_vs_id()
    # print(d2)
    # print(d1.__len__())