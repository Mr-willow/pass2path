from src.data.generate_path import find_med_backtrace
import matplotlib.pyplot as plt

def get_path_length_statistics(filename):
    length_dict = {}
    nums = 0
    with open(filename, 'r') as f:
        line = f.readline()

        while line:
            line = line.split('\t')
            pass1, pass2 = line[1], line[2]
            ed, path = find_med_backtrace(pass1, pass2)
            if ed == 0:
                line = f.readline()
                continue
            if len(path) not in length_dict:
                length_dict[len(path)] = 1
            else:
                length_dict[len(path)] += 1

            nums += 1
            line = f.readline()

    for k in length_dict:
        length_dict[k] /= nums

    x = sorted(length_dict.keys())
    y = [length_dict[i] for i in x]
    plt.figure()
    plt.title(filename.split('\\')[-1])
    plt.bar(x, y)
    plt.show()

    return length_dict


if __name__ == '__main__':
    get_path_length_statistics(r'E:\PycharmProjects\Tensorflow2\src\data\csdn_dodonew_reuse_uniq.txt')
    get_path_length_statistics(r'E:\PycharmProjects\Tensorflow2\src\data\12306_dodonew_reuse_uniq.txt')
    get_path_length_statistics(r'E:\PycharmProjects\Tensorflow2\src\data\csdn_126_reuse.txt')
    get_path_length_statistics(r'E:\PycharmProjects\Tensorflow2\src\data\csdn_12306_reuse_uniq.txt')
