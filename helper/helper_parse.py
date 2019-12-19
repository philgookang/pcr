import random

def prase_data_by_ratio(dataset, number):
    key_list = list(dataset.keys())
    random.seed(500)
    validation_set_key = []
    validation_set = {}
    test_set_key = []
    test_set = {}

    while len(test_set_key) < number:
        index = random.randint(0, (len(key_list) - 1))
        key = key_list[index]
        if key not in test_set_key and key not in validation_set_key:
            test_set_key.append(key)

    while len(validation_set_key) < number:
        index = random.randint(0, (len(key_list) - 1))
        key = key_list[index]
        if key not in test_set_key and key not in validation_set_key:
            validation_set_key.append(key)

    for k in validation_set_key:
        validation_set[k] = dataset[k]
        del dataset[k]

    for k in test_set_key:
        test_set[k] = dataset[k]
        del dataset[k]

    return dataset, validation_set, test_set
