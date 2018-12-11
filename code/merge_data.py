from glob import glob
import sys


if input('Merging files matching {} into {}. Confirm? (y/n) '.format(sys.argv[1], sys.argv[2])).strip().lower() == 'y':
	out = ''

	for in_file in glob(sys.argv[1] + '*'):
	    with open(in_file) as f:
	        out += ''.join(f.readlines()[1:])

	with open(sys.argv[2], 'w') as f:
	    f.write(out)
else:
	print('Cancelled.')
