import random

def prase_data_by_ratio(dataset, number):
    key_list = list(dataset.keys())
    random.seed(50)
    test_set_key = []
    validation_set_key = []
    test_set = {}
    validation_set = {}

    while len(test_set_key) < number:
        index = random.randint(0, (len(key_list) - 1))
        if key_list[index] not in test_set_key and key_list[index] not in validation_set_key:
            test_set_key.append(key_list[index])

    while len(validation_set_key) < number:
        index = random.randint(0, (len(key_list) - 1))
        if key_list[index] not in test_set_key and key_list[index] not in validation_set_key:
            validation_set_key.append(key_list[index])

    for k in test_set_key:
        test_set[k] = dataset[k]
        del dataset[k]

    for k in validation_set_key:
        validation_set[k] = dataset[k]
        del dataset[k]

    return dataset, test_set, validation_set
