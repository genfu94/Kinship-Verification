import os
import shutil
from . import data_ingestion_utils

DATA_SOURCES = [
    # {
    #     'url': "http://www.kinfacew.com/dataset/KinFaceW-I.zip",
    #     'filename': "KinFaceW-I.zip",
    #     'source': data_ingestion_utils.DownloadSource.HTTP
    # },
    # {
    #     'url': "http://www.kinfacew.com/dataset/KinFaceW-II.zip",
    #     'filename': "KinFaceW-II.zip",
    #     'source': data_ingestion_utils.DownloadSource.HTTP
    # },
    # {
    #     'url': "http://www1.ece.neu.edu/~yunfu/research/Kinface/KinFace_V2.zip",
    #     'filename': "KinFace_V2.zip",
    #     'source': data_ingestion_utils.DownloadSource.HTTP
    # },
    {
        'url': "https://drive.google.com/file/d/1qhxX4YrJRXvxrybxQo0yGhkwUdmen-js/view",
        'filename': "TSKinFace_Data.zip",
        'source': data_ingestion_utils.DownloadSource.GOOGLE_DRIVE
    },
    # {
    #     'url': "http://chenlab.ece.cornell.edu/projects/KinshipClassification/Family101_150x120.zip",
    #     'filename': "Family101_150x120.zip",
    #     'source': data_ingestion_utils.DownloadSource.HTTP
    # },
    # {
    #     'url': "http://chenlab.ece.cornell.edu/projects/KinshipVerification/KinshipVerification.zip",
    #     'filename': "KinshipVerification.zip",
    #     'source': data_ingestion_utils.DownloadSource.HTTP
    # }
]


def download_data(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for data_source in DATA_SOURCES:
        save_path = os.path.join(output_dir, data_source['filename'])

        if os.path.exists(save_path.replace('.zip', '')):
            print(f"Data from {data_source['url']} already exists. Skipping download")
            continue

        data_ingestion_utils.retrieve_data_from_url(data_source['url'], save_path, data_source['source'])
        shutil.unpack_archive(save_path, output_dir)
        os.remove(save_path)

    shutil.rmtree(os.path.join(output_dir, '__MACOSX'), ignore_errors = True)