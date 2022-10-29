from enum import Enum
from tqdm import tqdm
import requests
import gdown
import os


class DownloadSource(Enum):
	HTTP = 1
	GOOGLE_DRIVE = 2

# TODO: make interfaces for these download methods!
def _retrieve_file_from_url(url: str, filepath: str, show_progres: bool = True):
	with requests.get(url, stream=True) as r:
		r.raise_for_status()
		with open(filepath, 'wb') as f:
			pbar = tqdm(total=int(r.headers['Content-Length']), unit = 'B', unit_scale = True, disable = not show_progres)
			for chunk in r.iter_content(chunk_size=8192):
				if chunk:  # filter out keep-alive new chunks
					f.write(chunk)
					pbar.update(len(chunk))


# TODO: check why this takes a lot of time before start downloading
def _retrieve_file_from_google_drive(url: str, filepath: str, show_progress: bool = True):
	gdown.download(url, filepath, quiet=False, fuzzy=show_progress)

downloader_methods = {
	DownloadSource.HTTP: _retrieve_file_from_url,
	DownloadSource.GOOGLE_DRIVE: _retrieve_file_from_google_drive
}

def retrieve_data_from_url(url: str, filepath: str, source: DownloadSource, show_progress: bool = True):
	if source not in downloader_methods:
		raise("Download from the specified source is not implemented yet.")

	downloader = downloader_methods[source]

	if os.path.exists(filepath):
		print(f"The resource {filepath} already exists. Skipping download.")
		return

	downloader(url, filepath, show_progress)
