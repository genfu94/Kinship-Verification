from tqdm import tqdm
import requests
import gdown


def retrieve_file_from_url(url: str, filepath: str, show_progres: bool = True):
	with requests.get(url, stream=True) as r:
		r.raise_for_status()
		with open(filepath, 'wb') as f:
			pbar = tqdm(total=int(r.headers['Content-Length']), unit = 'B', unit_scale = True, disable = not show_progres)
			for chunk in r.iter_content(chunk_size=8192):
				if chunk:  # filter out keep-alive new chunks
					f.write(chunk)
					pbar.update(len(chunk))


# TODO: check why this takes a lot of time before start downloading
def retrieve_file_from_google_drive(url: str, filepath: str, show_progress: bool = True):
	gdown.download(url, filepath, quiet=False, fuzzy=show_progress)