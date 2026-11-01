import re

with open('comp_stack.py', 'r', encoding='utf-8') as f:
    text = f.read()

inputs = re.findall(r'/kaggle/input/[^\s\'\"\)]+', text)
print("Unique input paths in comp_stack.py:")
for inp in sorted(set(inputs)):
    print(" ", inp)

