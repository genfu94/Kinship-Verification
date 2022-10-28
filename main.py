import src.data_ingestion as data_ingestion

data_ingestion.retrieve_file_from_url("http://www.kinfacew.com/dataset/KinFaceW-I.zip", "data/original/KinFaceW-I.zip")
data_ingestion.retrieve_file_from_url("http://www.kinfacew.com/dataset/KinFaceW-I.zip", "data/original/KinFaceW-II.zip")
data_ingestion.retrieve_file_from_url("http://www1.ece.neu.edu/~yunfu/research/Kinface/KinFace_V2.zip", "data/original/KinFace_V2.zip")
data_ingestion.retrieve_file_from_google_drive("https://drive.google.com/file/d/1qhxX4YrJRXvxrybxQo0yGhkwUdmen-js/view", 'data/original/TSKinFace_Data.zip')
data_ingestion.retrieve_file_from_url("http://chenlab.ece.cornell.edu/projects/KinshipClassification/Family101_150x120.zip", 'data/original/Family101_150x120.zip')
data_ingestion.retrieve_file_from_url('http://chenlab.ece.cornell.edu/projects/KinshipVerification/KinshipVerification.zip', 'data/original/KinshipVerification.zip')