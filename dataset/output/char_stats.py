import os
from collections import Counter
from dataset.dataset import get_all_characters, get_radicals

all_characters = get_all_characters()
all_radicals = get_radicals()
num_characters = len(all_characters)
num_radicals = len(all_radicals)
character_counts = Counter()
radical_counts = Counter()

for label_file in os.listdir('labels'):
    if label_file.endswith('.txt'):
        with open(os.path.join('labels', label_file), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                char = parts[0]
                radical = parts[5]
                character_counts[char] += 1
                radical_counts[radical] += 1
                print(f"read file {label_file}")

total_characters = sum(character_counts.values())
total_radicals = sum(radical_counts.values())

character_frequencies = {char: count / total_characters for char, count in character_counts.items()}
radical_frequencies = {radical: count / total_radicals for radical, count in radical_counts.items()}

print(f"writing character statistics...")

with open('character_stats.csv', 'w', encoding='utf-8') as f:
    f.write('Character,Absolute Frequency,Relative Frequency,Weight\n')
    for char in all_characters:
        abs_freq = character_counts.get(char, 0)
        rel_freq = character_frequencies.get(char, 0.0)
        weight = total_characters/(num_characters*abs_freq)
        f.write(f'{char},{abs_freq},{rel_freq:.6f},{weight:.6f}\n')

print(f"writing radical statistics...")

with open('radical_stats.csv', 'w', encoding='utf-8') as f:
    f.write('Radical,Absolute Frequency,Relative Frequency,Weight\n')
    for radical in all_radicals:
        abs_freq = radical_counts.get(radical, 0)
        rel_freq = radical_frequencies.get(radical, 0.0)
        weight = total_radicals/(num_radicals*abs_freq)
        f.write(f'{radical},{abs_freq},{rel_freq:.6f},{weight:.6f}\n')

print(f"done.")
