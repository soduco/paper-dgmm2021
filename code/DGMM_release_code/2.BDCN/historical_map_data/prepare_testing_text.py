import os
from pathlib import Path

if __name__ == '__main__':
	# Write training files
	dir = str(Path(os.getcwd()))
		
	# Write testing files
	dir = str(Path(os.getcwd()))
	test_dir = dir + '/test'
	
	test_files = os.listdir(test_dir)
	with open(os.path.join(dir, 'test.lst'),'w+') as out:
		for t in test_files:
			out.write('{} {}\n'.format('test/'+t, 'test/'+t))
