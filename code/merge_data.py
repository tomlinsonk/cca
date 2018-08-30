from glob import glob


out = ''

for in_file in glob('*.csv'):
    with open(in_file) as f:
        out += ''.join(f.readlines()[1:])

with open('128_moore_diff.csv', 'w') as f:
    f.write(out)