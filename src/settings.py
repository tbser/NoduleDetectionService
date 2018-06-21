import os
# COMPUTER_NAME = os.environ['COMPUTERNAME']
# print("Computer: ", os.uname()[1])

global log
log = None
TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 1
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = "/opt/data/deeplearning/"
BASE_DIR = "/opt/data/deeplearning/"
# EXTRA_DATA_DIR = BASE_DIR + "resources/"
WORKING_DIR = os.getcwd() + "/"
print("Current working directory is ", WORKING_DIR)
EXTRA_DATA_DIR = WORKING_DIR + "resources/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "kaggle/dicom/"
LUNA16_RAW_SRC_DIR = BASE_DIR + "luna/"

NDSB3_EXTRACTED_IMAGE_DIR = WORKING_DIR + "ndsb3_extracted_images1/"
LUNA16_EXTRACTED_IMAGE_DIR = WORKING_DIR + "luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR = WORKING_DIR + "ndsb3_nodule_predictions/"

# HOSPITAL_DICOM_DIR = BASE_DIR + "hospital/"
HOSPITAL_DICOM_DIR = "D:/ZhongShan-DICOM/第二批-1-2/"
HOSPITAL_EXTRACTED_IMAGE_DIR = WORKING_DIR + "hospital_extracted_images/"
HOSPITAL_NODULE_DETECTION_DIR = WORKING_DIR + "hospital_nodule_predictions/positive/"
HOSPITAL_MANUAL_ANNOTATION_DIR = EXTRA_DATA_DIR + "hospital_manual_annotation/"

# dicom for prediction
# INCOMING_DICOM_DIR = '/home/meditool/windows-share/dicom/'
# INCOMING_EXTRACTED_IMAGE_DIR = WORKING_DIR +'extraced_images/'
# NODULE_DETECTION_DIR = "/home/meditool/windows-share/predict/"
# TRAINED_MODEL_3DCNN = '../models/model_luna16_full__fs_best.hd5'

PREDICTION_DICOM_DIR = '/home/meditool/windows-share/prediction_dicom/'
PREDICTION_EXTRACTED_IMAGE_DIR = WORKING_DIR + 'prediction_extracted_images/'

MANUAL_ANNOTATION_DICOM_DIR = "/home/meditool/windows-share/manualanno_dicom/"
MANUAL_ANNOTATION_EXTRACTED_IMAGE_DIR = WORKING_DIR + 'manualanno_extracted_images/'

NODULE_DETECTION_DIR = "/home/meditool/windows-share/predict/"
TRAINED_MODEL_3DCNN = '../models/model_luna16_full__fs_best.hd5'
TRAINED_MODEL_DIR = "../models/"
TRAINED_WORKING_DIR = "../workdir/"

ANNOTATION_MANUAL_DIR = "/home/meditool/windows-share/manual_annotation/"
ANNOTATION_CHECKED_DIR = "/home/meditool/windows-share/checked_annotation/"
ANNOTATION_POST_FORMAT_DIR = "/home/meditool/windows-share/post_format_annotation/"

CUBE_IMAGE_DIR = WORKING_DIR + "generated_traindata/"
