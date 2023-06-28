def change_m(line):
    if 'K'in line:
        tmp = 1000
    elif 'M' in line:
        tmp = 1000000
    line = line.split('.')[0]
    res = int(line) * tmp
    return res


def entropy(url):
    string = url.replace('NotGiven', '')
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
    return entropy


def is_ip(url):
    string = url
    flag = False
    if ("." in string):
        elements_array = string.strip().split(".")
        if(len(elements_array) == 4):
            for i in elements_array:
                if (i.isnumeric() and int(i)>=0 and int(i)<=255):
                    flag=True
                else:
                    flag=False
    if flag:
        return 1 
    else:
        return 0


def num_digits(url):
    digits = [i for i in url if i.isdigit()]
    return len(digits)


def url_length(url):
    return len(url)


def num_params(url):
    params = url.split('.')
    return len(params) - 1


def num_fragments(url):
    fragments = url.split('-')
    return len(fragments)
