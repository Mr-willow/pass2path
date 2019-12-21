from data.generate_data import id2path, num2char
from data.word2keyboard import switch_from_keyboard_list

def path2pass(source, path):
    '''

    :param source: list of pass id
    :param path: list of path id
    :return: new pass
    '''
    for i in range(len(source)):
        source[i] = num2char[source[i]]
    for i in range(len(path)):
        step = id2path[path[i]]
        if step[0] == 'd':
            source[step[2]] = ''
        elif step[0] == 'i':
            source.insert(step[2], step[1])
        elif step[0] == 's':
            source[step[2]] = step[1]

    return switch_from_keyboard_list(source)


if __name__ == '__main__':
    s = [2, 3, 4, 11]
    p = [1, 10, 120]
    print(id2path[1], id2path[10], id2path[120])
    print(path2pass(s, p))
