#!/usr/bin/env python
# -*- coding:utf-8 -*-
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
import src.preprocess_hospital as preprocess
import src.preprocess_make_train_cubes as make_cubes
import src.settings as settings
import src.helpers as helpers
from src.ThreeDCNN import ThreeDCNN
import os
import pandas
import zipfile
import json, simplejson
import time, glob

app = Flask(__name__)
api = Api(app)

# Run with:
#
#   $ gunicorn prediction_service_update:app
#

parser = reqparse.RequestParser()
parser.add_argument('patient_id')
# parser.add_argument('dicom_folder')

logger = helpers.getlogger("prediction_service_update")
CUBE_SIZE = 32
cnnmodel = ThreeDCNN(logger)
t = time.localtime()


class Predict(Resource):
    def get(self, patient_id):
        try:
            # data = parser.parse_args()
            # if data['patient_id'] and data['dicom_folder']:
            # if data['patient_id']:
            if patient_id is not None:
                logger.info("extract the DICOM zip of {}.".format(patient_id))
                dicomzipfile = settings.PREDICTION_DICOM_DIR + patient_id + ".zip"
                if not os.path.exists(dicomzipfile):
                    logger.error("DICOM zip file {} does not exist.".format(dicomzipfile))
                    return "DICOM zip file does not exist.", 404
                with zipfile.ZipFile(dicomzipfile, 'r') as zip_ref:
                    extractfolder = settings.PREDICTION_DICOM_DIR + patient_id + "/"
                    if not os.path.exists(extractfolder):
                        os.makedirs(extractfolder)
                    zip_ref.extractall(extractfolder)

                logger.info("extract the DICOM of {} to images.".format(patient_id))
                preprocess.extract_dicom_images(settings.PREDICTION_DICOM_DIR,
                                                settings.PREDICTION_EXTRACTED_IMAGE_DIR,
                                                clean_targetdir_first=True,
                                                only_patient_id=patient_id)

                logger.info("3DCNN predict.")
                cnnmodel.generate_model((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), dropout=True, batchnormalization=False, load_weight_path=settings.TRAINED_MODEL_3DCNN)
                cnnmodel.predict_patient(patient_id, settings.PREDICTION_EXTRACTED_IMAGE_DIR, settings.NODULE_DETECTION_DIR)
                result_csv_file = settings.NODULE_DETECTION_DIR + patient_id + '.csv'
                if not os.path.exists(result_csv_file):
                    return "result csv is not generated", 404

                df = pandas.read_csv(result_csv_file)
                json = df.to_json(orient='records')
                logger.info("The predict result for {}:".format(patient_id))
                logger.info(json)
                return json
            else:
                return "missing parameters", 404
        except Exception as ex:
            return str(ex), 500


class SaveAnno(Resource):
    # 查看之前保存的Annotation  注:要用之前前端传过来的格式
    def get(self, patient_id):
        try:
            if patient_id is not None:
                logger.info("get the saved annotation of {}.".format(patient_id))
                # anno_dir = settings.ANNOTATION_DIR  # "/home/meditool/windows-share/annotation/"
                # dirname: /home/meditool/windows-share/annotation/
                #          /home/meditool/windows-share/annotation/2018_4
                # subdirlist: ['2018_4'] []
                # filelist: [] ['001.csv']
                # anno_file = ""
                # for dirname, subdirlist, filelist in os.walk(anno_dir):
                #     for filename in filelist:
                #         if filename.lower().endswith(".csv"):
                #             if patient_id == filename.replace(".csv", ""):
                #                 anno_file = os.path.join(dirname, filename)
                #             else:
                #                 continue
                # df = pandas.read_csv(anno_file)
                # json_str = df.to_json(orient='records')
                # logger.info("The annotation for {}:".format(patient_id))
                # logger.info(json_str)

                post_dir = settings.ANNOTATION_POST_FORMAT_DIR
                post_file = glob.glob(post_dir + "*.txt")
                post_stri = ""
                for file_path in post_file:                     # /../../file.txt
                    dirpath, filename = os.path.split(file_path)   # /../..   file.txt
                    if filename.lower().endswith(".txt"):
                        if patient_id == filename.replace(".txt", ""):
                            with open(file_path, 'r') as f:
                                post_stri = f.readlines()
                        else:
                            continue
                json_str = json.dumps(post_stri)
                logger.info("The annotation for {}:".format(patient_id))
                logger.info(json_str)
                return json_str
            else:
                return "missing parameters", 404
        except Exception as ex:
            return str(ex), 500

    # 增  增加新的人工标注的Annotation
    def post(self, patient_id):
        try:
            if patient_id is not None:
                # post {"data": "annotation"}
                json_str = request.data.decode('utf-8')

                # used to save the annotation posted by front-end, keeping the format
                post_saved_dir = settings.ANNOTATION_POST_FORMAT_DIR
                if not os.path.exists(post_saved_dir):
                    os.makedirs(post_saved_dir)
                post_csv_path = post_saved_dir + patient_id + '.txt'
                with open(post_csv_path, 'w') as f:
                    f.writelines(json_str)

                # used to train
                dst_dir = settings.ANNOTATION_MANUAL_DIR + str(t.tm_year) + "_" + str(t.tm_mon) + '/'
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                csv_target_path = dst_dir + patient_id + '.csv'

                json_dict = simplejson.loads(json_str)  # convert json to dictionary format
                anno_str = json_dict["data"]
                parts = anno_str.split(",;,")
                # logger.info(parts)
                annotations_csv = []
                for i in range(len(parts) - 1):  # parts[len(parts)-1]为空
                    anno_dict = eval(parts[i])   # convert string '{key:value}' to dictionary {key:value}
                    # "Pos_1": [114.33069525070366,227.43840743467692,79]
                    # "Diameter":52.23627789894766,
                    # "center_perc":[0.06681079608359353,0.07710518839281738,0.016853932584269662],
                    # "Slice_number":3,"IsNodule":1,"strSlice":78,"endSlice":80,
                    # "originScale":1.1812373399734497,"type":"circle","nodeNumber":2
                    type_anno = anno_dict["type"]
                    if type_anno == 'length' or type_anno == 'square' or type_anno == 'circle':
                        pos_1, pos_2, diameter = anno_dict["Pos_1"], anno_dict["Pos_2"], anno_dict["Diameter"]
                        center_perc = anno_dict["center_perc"]
                        slice_number, is_nodule = anno_dict["Slice_number"], anno_dict["IsNodule"]
                        start_slice, end_slice = anno_dict["strSlice"], anno_dict["endSlice"]
                        origin_scale, node_number = anno_dict["originScale"], anno_dict["nodeNumber"]

                        pos_1_x, pos_1_y, pos_1_z = pos_1[0], pos_1[1], pos_1[2]
                        pos_2_x, pos_2_y, pos_2_z = pos_2[0], pos_2[1], pos_2[2]
                        coordX, coordY, coordZ = (pos_1_x + pos_2_x) / 2, (pos_1_y + pos_2_y) / 2, pos_1_z
                        diameter_mm = diameter
                        annotations_csv_line = [patient_id, coordX, coordY, coordZ, diameter_mm]
                        annotations_csv.append(annotations_csv_line)
                    else:
                        continue

                df = pandas.DataFrame(annotations_csv, columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
                df.to_csv(csv_target_path, index=False)
                logger.info("the annotations in the post are saved in: {} ".format(post_csv_path))
                logger.info("the converted annotations for training are saved in: {} ".format(csv_target_path))
                json_converted = df.to_json(orient='records')
                logger.info(json_converted)
                return json_str    # "The new annotation has been saved."
            else:
                return "missing parameters", 404
        except Exception as ex:
            return str(ex), 500

    # 改  修改模型预测结果得到的Annotation更新   注:前端只是用来人工标注。此功能针对platform
    def put(self, patient_id):
        try:
            if patient_id is not None:
                dst_dir = settings.ANNOTATION_CHECKED_DIR
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)

                csv_target_path = dst_dir + patient_id + '_' + str(t.tm_year)
                for i in (t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min):
                    if i < 10:
                        csv_target_path += "0" + str(i)
                    else:
                        csv_target_path += str(i)
                csv_target_path += '.csv'

                # post {"data": "annotation"}
                json_str = request.data.decode('utf-8')
                json_dict = simplejson.loads(json_str)  # convert json to dictionary format
                anno_str = json_dict["data"]
                parts = anno_str.split(";")
                # logger.info(parts)
                annotations_csv = []
                for i in range(len(parts) - 1):  # parts[len(parts)-1]为空
                    anno_dict = eval(parts[i])  # convert string '{key:value}' to dictionary {key:value}
                    type_anno = anno_dict["type"]
                    if type_anno == 'length' or type_anno == 'square' or type_anno == 'circle':
                        pos_1, pos_2, diameter = anno_dict["Pos_1"], anno_dict["Pos_2"], anno_dict["Diameter"]
                        center_perc = anno_dict["center_perc"]
                        slice_number, is_nodule = anno_dict["Slice_number"], anno_dict["IsNodule"]
                        start_slice, end_slice = anno_dict["strSlice"], anno_dict["endSlice"]
                        origin_scale, node_number = anno_dict["originScale"], anno_dict["nodeNumber"]

                        pos_1_x, pos_1_y, pos_1_z = pos_1[0], pos_1[1], pos_1[2]
                        pos_2_x, pos_2_y, pos_2_z = pos_2[0], pos_2[1], pos_2[2]
                        coordX, coordY, coordZ = (pos_1_x + pos_2_x) / 2, (pos_1_y + pos_2_y) / 2, pos_1_z
                        diameter_mm = diameter
                        annotations_csv_line = [patient_id, coordX, coordY, coordZ, diameter_mm]
                        annotations_csv.append(annotations_csv_line)
                    else:
                        continue
                df = pandas.DataFrame(annotations_csv,
                                      columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
                df.to_csv(csv_target_path, index=False)

                logger.info("the annotations in the post are saved in: {} ".format(csv_target_path))
                json = df.to_json(orient='records')
                logger.info(json)
                return json   # "The updated annotation has been saved."
            else:
                return "missing parameters", 404
        except Exception as ex:
            return str(ex), 500


class Train(Resource):
    # 告诉服务器训练新的模型
    def post(self):
        try:
            model_name = str(t.tm_year)
            for i in (t.tm_mon, t.tm_mday, t.tm_hour):
                if i < 10:
                    model_name += "0" + str(i)
                else:
                    model_name += str(i)
            # train_data_generator = "new_data"
            # train_data_size = "the number of train data"
            # validate_data_generator = "validate_data"
            # validate_data_size = "the number of validate data"
            working_dir = settings.TRAINED_WORKING_DIR + model_name + '/'
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)

            models_dir = settings.TRAINED_MODEL_DIR
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            # 1、解压 人工标注 manualanno_dicom 下的所有zip
            patient_id = ""
            dicomzipfile = settings.MANUAL_ANNOTATION_DICOM_DIR + patient_id + ".zip"
            if not os.path.exists(dicomzipfile):
                logger.error("DICOM zip file {} does not exist.".format(dicomzipfile))
                return "DICOM zip file does not exist.", 404
            with zipfile.ZipFile(dicomzipfile, 'r') as zip_ref:
                extractfolder = settings.MANUAL_ANNOTATION_DICOM_DIR + patient_id + "/"
                if not os.path.exists(extractfolder):
                    os.makedirs(extractfolder)
                zip_ref.extractall(extractfolder)

            # 2、对解压后的dicom文件 进行预处理 生成 *_i.png *_m.png
            preprocess.extract_dicom_images(settings.MANUAL_ANNOTATION_DICOM_DIR,
                                            settings.MANUAL_ANNOTATION_EXTRACTED_IMAGE_DIR,
                                            clean_targetdir_first=True,
                                            only_patient_id=patient_id)

            # 3、对上面得到的png 根据manual_annotation 来make cubes
            make_cubes.make_pos_annotation_manual_images(settings.MANUAL_ANNOTATION_EXTRACTED_IMAGE_DIR,
                                                         settings.CUBE_IMAGE_DIR+'manualanno_train_cubes/',
                                                         clean_targetdir_first=True,
                                                         only_patient_id=patient_id)
            # 4、加载权重, 训练模型
            logger.info("Train the model.")
            # =====加载的模型权重load_weight_path: 最近一次训练的模型?
            cnnmodel.generate_model((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), dropout=True, batchnormalization=False,
                                    load_weight_path=settings.TRAINED_MODEL_3DCNN)
            cnnmodel.train(model_name, train_epoch_save_folder=working_dir,
                           train_model_save_folder=models_dir, epoch_number=12, batch_size=16)

            return "The new model is being trained. About 17 hours will be needed."
        except Exception as ex:
            return str(ex), 500

# api.add_resource(Predict, '/predict')
api.add_resource(Predict, '/predict/<patient_id>')
api.add_resource(SaveAnno, '/save/<patient_id>')
api.add_resource(Train, '/train')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
    # app.run()
