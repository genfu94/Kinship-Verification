from tqdm import tqdm
import requests


def retrieve_file_from_url(url: str, filepath: str, show_progres: bool = True):
	with requests.get(url, stream=True) as r:
		r.raise_for_status()
		with open(filepath, 'wb') as f:
			pbar = tqdm(total=int(r.headers['Content-Length']), unit = 'B', unit_scale = True, unit_divisor = 1024, disable = not show_progres)
			for chunk in r.iter_content(chunk_size=8192):
				if chunk:  # filter out keep-alive new chunks
					f.write(chunk)
					pbar.update(len(chunk))


retrieve_file_from_url("http://www.kinfacew.com/dataset/KinFaceW-I.zip", "KinFaceW-I.zip")
retrieve_file_from_url("http://www1.ece.neu.edu/~yunfu/research/Kinface/KinFace_V2.zip", "KinFace_V2.zip")