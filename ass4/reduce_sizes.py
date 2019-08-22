STUDENT = {'name': 'STEVE GUTFREUND_YOEL JASNER',
           'ID': '342873791_204380992'}


NO_LABEL = "\"gold_label\": \"-\""

import random


def create_shorter_version(data, output_file, size):
    random.shuffle(data)
    o = open(output_file, 'w')

    for line in data[:size]:
        o.write(line)

    o.close()


def load(filename):
    print("\nLoading file...")

    data = []

    for line in open(filename):
        if NO_LABEL in line:
            continue

        data.append(line)

    print("{} pairs".format(len(data)))
    return data

percentage = 0.1

data_files = "snli_1.0_{}.jsonl"

TRAIN = load(data_files.format("train"))
TRAIN_SIZE = int(len(TRAIN) * percentage)
create_shorter_version(TRAIN, data_files.format("train_plan_C"), TRAIN_SIZE)

DEV = load(data_files.format("dev"))
DEV_SIZE = int(len(DEV) * percentage)
create_shorter_version(DEV, data_files.format("dev_plan_C"), DEV_SIZE)

TEST = load(data_files.format("test"))
TEST_SIZE = int(len(TEST) * percentage)
create_shorter_version(TEST, data_files.format("test_plan_C"), TEST_SIZE)
