from src.data_ingestion import download_data
from src.data_cleaning import KinFace_V2_Parser, KinFaceWParser, TSKinFaceParser

#download_data('data/original')
#kinface_v2_parser = KinFace_V2_Parser()
#kinface_v2_parser.parse('data/original/KinFace_V2')

#kinfacew_parser = KinFaceWParser()
#kinfacew_parser.parse('data/original/KinFaceW-I')
#kinfacew_parser.parse('data/original/KinFaceW-II')
tskinface_parser = TSKinFaceParser()
features = tskinface_parser.parse('data/original/TSKinFace_Data/TSKinFace_cropped')
