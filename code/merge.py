from glob import glob
import sys
import os

to_merge = sys.argv[1] + '*'
print(to_merge)

combined_dir = '/Users/kiran/Desktop/Projects/Research/cca/data/combined/'
save_name = combined_dir + sys.argv[1].split('/')[-1].strip('_') + '.csv'

out = ''

for in_file in glob(to_merge):
    with open(in_file) as f:
        out += ''.join(f.readlines()[1:])

print('Doing: {}'.format(save_name))
with open(save_name, 'w') as f:
    f.write(out)