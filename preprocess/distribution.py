import json

import matplotlib.pyplot as plt

with open('./data/dataset/train.json', 'r') as f:
    conll_data = json.load(f)

average_len, max_len = 0, -1
len_dict = {idx: 0 for idx in range(0, 115)}

for i, sample in enumerate(conll_data):
    sample_len = len(sample['sentence'].split())
    assert len(sample['sentence'].split()) == len(
        sample['tags'].split()), 'Fatal error !'  # check again
    average_len += sample_len
    if sample_len > max_len:
        max_len = sample_len
    len_dict[sample_len] += 1

average_len /= len(conll_data)

print(f'max_len: {max_len}')  # 113
print(f'average_len: {average_len:.2f}')  # 13.65

# get distribution of the length of sentences
plt.bar(list(len_dict.keys()), len_dict.values())
plt.xticks([idx for idx in range(0, 115, 10)])
plt.title('distribution of sentences length')
plt.show()
