import random
import csv

def generate_data(num):
    data_list = []
    for _ in range(num):
        data = []
        data.append(0)
        data.append(random.randint(200, 800))
        data.append(random.randint(100, 400) / 100)
        data.append(random.randint(1, 4))
        
        score = (data[1]-200) / 600 + (data[2]-1) / 3 + (data[3]-0.5) / 4
        score /= 3
        # rand_num = random.random()
        if score > 0.5:
            data[0] = 1
        data_list.append(data)
    return data_list

if __name__ == '__main__':
    train_list = generate_data(500)
    test_list = generate_data(100)
    # print(train_list)
    # print(test_list)
    with open('train.csv', 'w', newline='') as train_file:
        writer = csv.writer(train_file, delimiter=',')
        writer.writerow(['admit', 'gre', 'gpa', 'rank'])
        for row in train_list:
            writer.writerow(row)
    with open('test.csv', 'w', newline='') as test_file:
        writer = csv.writer(test_file, delimiter=',')
        writer.writerow(['admit', 'gre', 'gpa', 'rank'])
        for row in test_list:
            writer.writerow(row)
    