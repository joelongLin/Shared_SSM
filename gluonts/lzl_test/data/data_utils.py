#判断 当前input　是否在all列表中，判断标准是第dim维度
def whetherInputInList(input , all , dim):
    if len(all) == 0:
        return False
    for value in all:
        if (value[dim] == input[dim]).all():
            return True

    return False