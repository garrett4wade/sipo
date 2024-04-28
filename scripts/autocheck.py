import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_landmarks", type=int, default=3)
args = parser.parse_args()
n_landmarks = args.n_landmarks

def chk_iter():
	f_iter = open('iter.txt', 'r')
	lines = f_iter.readlines()
	f_iter.close()

	it = 0
	archive = []
	for line in lines:
		if line.find('Eval') != -1:
			cut = line[line.find('Nearest landmark') + 18: -3]
			landmarks = cut.split(' ')
			landmarks = list(filter(None, landmarks))
			landmarks = [float(x) for x in landmarks]
			idx = landmarks.index(max(landmarks))
			if landmarks[idx] < 1e-5:
				idx = -1
			print(f'Iteration {it}: {landmarks} position: {idx}')
			archive.append(landmarks.index(max(landmarks)))
			it += 1
	archive = sorted(archive)
	if archive == list(range(it)):
		print('iter check passed')
	else:
		print('iter check failed')

def chk_pbt():
	f_pbt = open('pbt.txt', 'r')
	lines = f_pbt.readlines()
	f_pbt.close()

	archive = [0] * n_landmarks
	for line in lines:
		if line.find('Nearest landmark') != -1:
			cut = line[line.find('Nearest landmark') + 18: -3]
			landmarks = cut.split(' ')
			landmarks = list(filter(None, landmarks))
			landmarks = [float(x) for x in landmarks]
			idx = landmarks.index(max(landmarks))
			if landmarks[idx] < 1e-5:
				idx = -1
			
			archive_idx = int(line[line.find('Rank') + 5: ].split(' ')[0])
			archive[archive_idx] = idx
			# print(f'Archive {archive_idx}: {landmarks} position: {landmarks.index(max(landmarks))}')
	print('pbt result:', archive)
	archive = sorted(archive) 
	if archive == list(range(n_landmarks)):
		print('pbt check passed')
	else:
		print('pbt check failed')

chk_iter()
chk_pbt()