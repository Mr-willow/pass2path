
def get_start_and_end_symbol(filename):
    total_num = 0
    s_number = 0
    s_word = 0
    s_character = 0
    e_number = 0
    e_word = 0
    e_character = 0

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            line = line.split('\t')
            pass1, pass2 = line[1], line[2]
            if 48 <= ord(pass1[0]) <= 57:
                s_number += 1
            elif (65 <= ord(pass1[0]) <= 90) or (97 <= ord(pass1[0]) <= 122):
                s_word += 1
            else:
                s_character += 1

            if 48 <= ord(pass2[0]) <= 57:
                s_number += 1
            elif (65 <= ord(pass2[0]) <= 90) or (97 <= ord(pass2[0]) <= 122):
                s_word += 1
            else:
                s_character += 1

            if 48 <= ord(pass1[-1]) <= 57:
                e_number += 1
            elif (65 <= ord(pass1[-1]) <= 90) or (97 <= ord(pass1[-1]) <= 122):
                e_word += 1
            else:
                e_character += 1

            if 48 <= ord(pass2[-1]) <= 57:
                e_number += 1
            elif (65 <= ord(pass2[-1]) <= 90) or (97 <= ord(pass2[-1]) <= 122):
                e_word += 1
            else:
                e_character += 1

            total_num += 2

            line = f.readline()

    return {'d': s_number / total_num, 'l': s_word / total_num, 's': s_character / total_num}, \
           {'d': e_number / total_num, 'l': e_word / total_num, 's': e_character / total_num}


if __name__ == '__main__':
    print(get_start_and_end_symbol(r'E:\PycharmProjects\Tensorflow2\src\data\12306_dodonew_reuse_uniq.txt'))