import os
import re
from datetime import datetime
import csv


# Define the folder containing the log files
log_folder = '.'

# Initialize the loss lists
rpn_localization_loss = []
rpn_classification_loss = []
head_localization_loss = []
super_classification_loss = []
sub_classification_loss = []

# Define the regex pattern to match the loss values
pattern = re.compile(
    r'RPN Localization Loss\s*:\s*(\d+\.\d+)\s*'
    r'RPN Classification Loss\s*:\s*(\d+\.\d+)\s*'
    r'Head Localization Loss\s*:\s*(\d+\.\d+)\s*'
    r'Super Classification Loss\s*:\s*(\d+\.\d+)\s*'
    r'Sub Classification Loss\s*:\s*(\d+\.\d+)'
)

# Get the list of log files sorted by date
log_files = sorted(
    [f for f in os.listdir(log_folder) if f.startswith('log_') and f.endswith('.txt')],
    key=lambda x: datetime.strptime(x, 'log_%Y-%m-%d_%H-%M-%S.txt')
)

# Process each log file
for log_file in log_files:
    with open(os.path.join(log_folder, log_file), 'r') as file:
        print(f"Reading {log_file}...")
        content = file.read()
        matches = pattern.findall(content)
        for match in matches:
            rpn_localization_loss.append(float(match[0]))
            rpn_classification_loss.append(float(match[1]))
            head_localization_loss.append(float(match[2]))
            super_classification_loss.append(float(match[3]))
            sub_classification_loss.append(float(match[4]))

# Save the losses to a CSV file
with open('losses.csv', 'w', newline='') as csvfile:
    print(f"Writing csv...")
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['RPN Localization Loss', 'RPN Classification Loss', 'Head Localization Loss', 'Super Classification Loss', 'Sub Classification Loss'])
    for i in range(len(rpn_localization_loss)):
        csvwriter.writerow([
            rpn_localization_loss[i],
            rpn_classification_loss[i],
            head_localization_loss[i],
            super_classification_loss[i],
            sub_classification_loss[i]
        ])
