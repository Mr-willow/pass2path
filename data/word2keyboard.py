
caps = chr(12)
shift = chr(11)


def switch2keyboard_list(source):
    caps = chr(12)
    shift = chr(11)
    keyboard_list = []
    shift_dict = {126: 96, 33: 49, 64: 50, 35: 51, 36: 52, 37: 53, 94: 54, 38: 55, 42: 56, 40: 57, 41: 48, 95: 45, 43: 61,
                  123: 91, 125: 93, 124: 92, 58: 59, 34: 39, 60: 44, 62: 46, 63: 47}
    is_upper = False
    for i in range(len(source)):
        if ord(source[i]) in shift_dict:
            keyboard_list.append(shift)
            keyboard_list.append(chr(shift_dict[ord(source[i])]))
            continue
        if is_upper:
            if 97 <= ord(source[i]) <= 122:  # lower letter
                keyboard_list.append(caps)
                keyboard_list.append(source[i])
                is_upper = False
            elif 65 <= ord(source[i]) <= 90:  # upper letter
                keyboard_list.append(chr(ord(source[i])+32))
            else:
                keyboard_list.append(source[i])
        else:
            if 65 <= ord(source[i]) <= 90:
                keyboard_list.append(caps)
                keyboard_list.append(chr(ord(source[i])+32))
                is_upper = True
            else:
                keyboard_list.append(source[i])
    return keyboard_list


def switch_from_keyboard_list(keyboard_list):
    word = ''
    shift_dict = {96: 126, 49: 33, 50: 64, 51: 35, 52: 36, 53: 37, 54: 94, 55: 38, 56: 42, 57: 40, 48: 41, 45: 95,
                  61: 43, 91: 123, 93: 125, 92: 124, 59: 58, 39: 34, 44: 60, 46: 62, 47: 63}
    is_upper = False
    is_shift = False
    for i in range(len(keyboard_list)):
        if is_shift:
            is_shift = False
            continue
        if keyboard_list[i] == caps:
            is_upper = not is_upper
        elif keyboard_list[i] == shift:
            word += chr(shift_dict[ord(keyboard_list[i + 1])])
            is_shift = True
        elif 97 <= ord(keyboard_list[i]) <= 122:  # lower letter
            if is_upper:
                word += chr(ord(keyboard_list[i])-32)
            else:
                word += keyboard_list[i]
        else:
            word += keyboard_list[i]

    return word


if __name__ == "__main__":
    s = 'ASD123asd~!@...#$%^&*()_+{}|:"<>?'
    l = switch2keyboard_list(s)
    print(l)
    s1 = switch_from_keyboard_list(l)
    print(s1)







