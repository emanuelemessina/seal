import random
import matplotlib.pyplot as plt
# Find and print numbers that were not generated

# Generate random integers and count frequencies
num_samples = 35000
max_value = 5684
frequencies = [0] * (max_value + 1)

for _ in range(num_samples):
    rand_int = random.randint(0, max_value)
    frequencies[rand_int] += 1

missing_numbers = [i for i, freq in enumerate(frequencies) if freq == 0]
print(f"Missing numbers ({len(missing_numbers)}): {missing_numbers}")

min_freq = min(frequencies)
min_freq_int = frequencies.index(min_freq)

print(f"lowest frequency : {min_freq} (of {min_freq_int})")

max_freq = max(frequencies)
max_freq_int = frequencies.index(max_freq)

print(f"highest frequency : {max_freq} (of {max_freq_int})")

# Plot the frequencies
plt.figure(figsize=(12, 6))
plt.bar(range(max_value + 1), frequencies, width=1.0)
plt.xlabel('Integer Value')
plt.ylabel('Frequency')
plt.title('Frequency of Random Integers between 0 and 6000')

# Annotate the missing frequencies
for missing in missing_numbers:
    plt.annotate('', xy=(missing, 0), xytext=(missing, 50),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=0.01))

plt.show()
