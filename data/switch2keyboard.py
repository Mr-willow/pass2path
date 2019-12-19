
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
            if 97 <= ord(source[i]) <= 122:
                keyboard_list.append(caps)
                keyboard_list.append(source[i])
                is_upper = False
            elif 65 <= ord(source[i]) <= 90:
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


if __name__ == "__main__":
    print(switch2keyboard_list('ASD123asd~!@#$%^&*()_+{}|:"<>?'))





