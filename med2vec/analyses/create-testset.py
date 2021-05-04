import pickle

seqs = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15, 16, 10, 9, 17, 18],
    [19, 0, 20, 11, 21, 22, 23, 9, 10, 24, 25, 14, 26],
    [27, 28, 29, 30],
    [31, 32, 26, 25, 33, 34, 35, 36, 37],
    [-1],
    [38, 39, 40, 41],
    [42, 43, 44, 45, 46, 47, 48, 41]
]

input_list = []
total_codes = 0
for seq in seqs:
    for code in seq:
        if code != -1:
            input_list.append(code)

# 49
print(len(list(set(input_list))))

labels = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [10, 11, 12, 13, 8, 14, 15, 16, 17],
    [10, 2, 4, 16, 17, 18, 19, 20, 21, 7, 22, 8],
    [23, 24, 25, 26],
    [10, 18, 12, 27, 28, 29, 30, 19, 21],
    [-1],
    [31, 32, 18, 33],
    [32, 34, 35, 36, 37, 33, 38, 39]
]

outFile = 'test_2pat_data/'
#pickle.dump(seqs, open(outFile + 'seqs.pkl', 'wb'), -1)
#pickle.dump(labels, open(outFile + 'labels.pkl', 'wb'), -1)
