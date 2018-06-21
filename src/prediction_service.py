from flask import Flask
from flask_restful import Api, Resource, reqparse
import preprocess_hospital as preprocess
import settings, helpers
from ThreeDCNN import ThreeDCNN
import os
import pandas
import zipfile

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('patient_id')
# parser.add_argument('dicom_folder')

logger = helpers.getlogger("prediction_service")
CUBE_SIZE = 32
cnnmodel = ThreeDCNN(logger)


class Predict(Resource):
    def get(self, patient_id):
        try:
            #data = parser.parse_args()
            #if data['patient_id'] and data['dicom_folder']:
            #if data['patient_id']:
            if patient_id is not None:
                logger.info("extract the DICOM zip of {}.".format(patient_id))
                dicomzipfile = settings.INCOMING_DICOM_DIR + patient_id + ".zip"
                if not os.path.exists(dicomzipfile):
                    logger.error("DICOM zip file {} does not exist.".format(dicomzipfile))
                    return "DICOM zip file does not exist.", 404
                with zipfile.ZipFile(dicomzipfile, 'r') as zip_ref:
                    extractfolder = settings.INCOMING_DICOM_DIR + patient_id + "/"
                    if not os.path.exists(extractfolder):
                        os.makedirs(extractfolder)
                    zip_ref.extractall(extractfolder)

                logger.info("extract the DICOM of {} to images.".format(patient_id))
                preprocess.extract_dicom_images(settings.INCOMING_DICOM_DIR,
                                                settings.INCOMING_EXTRACTED_IMAGE_DIR,
                                                clean_targetdir_first=True,
                                                only_patient_id=patient_id)

                logger.info("3DCNN predict.")
                cnnmodel.generate_model((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), dropout=True, batchnormalization=False, load_weight_path=settings.TRAINED_MODEL_3DCNN)
                cnnmodel.predict_patient(patient_id, settings.INCOMING_EXTRACTED_IMAGE_DIR, settings.NODULE_DETECTION_DIR)
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

#api.add_resource(Predict, '/predict')
api.add_resource(Predict, '/predict/<patient_id>')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
    #app.run()

