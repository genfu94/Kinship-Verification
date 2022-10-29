import os
import shutil
import src.data_ingestion as data_ingestion

DATA_SOURCES = [
    {
        'url': "http://www.kinfacew.com/dataset/KinFaceW-I.zip",
        'filepath': "data/original/KinFaceW-I.zip",
        'source': data_ingestion.DownloadSource.HTTP
    },
    {
        'url': "http://www.kinfacew.com/dataset/KinFaceW-II.zip",
        'filepath': "data/original/KinFaceW-II.zip",
        'source': data_ingestion.DownloadSource.HTTP
    },
    {
        'url': "http://www1.ece.neu.edu/~yunfu/research/Kinface/KinFace_V2.zip",
        'filepath': "data/original/KinFace_V2.zip",
        'source': data_ingestion.DownloadSource.HTTP
    },
    {
        'url': "https://drive.google.com/file/d/1qhxX4YrJRXvxrybxQo0yGhkwUdmen-js/view",
        'filepath': "data/original/TSKinFace_Data.zip",
        'source': data_ingestion.DownloadSource.GOOGLE_DRIVE
    },
    {
        'url': "http://chenlab.ece.cornell.edu/projects/KinshipClassification/Family101_150x120.zip",
        'filepath': "data/original/Family101_150x120.zip",
        'source': data_ingestion.DownloadSource.HTTP
    },
    {
        'url': "http://chenlab.ece.cornell.edu/projects/KinshipVerification/KinshipVerification.zip",
        'filepath': "data/original/KinshipVerification.zip",
        'source': data_ingestion.DownloadSource.HTTP
    }
]


for data_source in DATA_SOURCES:
    data_ingestion.retrieve_data_from_url(**data_source)
