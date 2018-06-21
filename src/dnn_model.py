#!usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
import os
os.environ['PYTHONHASHSEED'] = '0'
from keras.callbacks import Callback
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
import numpy
import pandas
import src.helpers as helpers
import src.settings as settings
import glob, random, ntpath, shutil

numpy.random.seed(42)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)


class DNN_model(object):
    LEARN_RATE = 0.001
    TENSORBOARD_LOG_DIR = "tfb_log/"
    tensorboard_enabled = True
    PREDICT_STEP = 12
    CUBE_SIZE = 32
    MEAN_PIXEL_VALUE = 41
    P_TH = 0.6
    NEGS_PER_POS = 2

    def __init__(self):
        self.MODEL_SUMMARY_FILE = './model_summary.txt'
        self.logger = None
        self.model = None
        self.trained = False
        self.callbacks = []

    def writemodelsummary(self, s):
        with open(self.MODEL_SUMMARY_FILE, 'a') as f:
            f.write(s + '\n')
            print(s)

    def model_summary(self, model):
        open(self.MODEL_SUMMARY_FILE, "w")
        model.summary(print_fn=self.writemodelsummary)

    def step_decay(self, epoch):
        res = 0.001
        if epoch > 5:
            res = 0.0001
        if self.logger is not None:
            self.logger.info("learnrate: {0} epoch: {1}".format(res, epoch))
        return res

    def analysis_filename(self, file_name):
        # hostpitalmanual_CHEN-LEYAN_5_pos_0_4_1_pn.png
        # ndsb3manual_2d81a9e760a07b25903a8c4aeb444eca_1_pos_0_18_1_pn.png
        # 1.3.6.1.4.1.14519.5.2.1.6279.6001.254254303842550572473665729969_2945xpointx0_9_1_pos.png

        # 1.3.6.1.4.1.14519.5.2.1.6279.6001.315918264676377418120578391325_492_0_luna.png
        # 1.3.6.1.4.1.14519.5.2.1.6279.6001.707218743153927597786179232739_119_0_edge.png
        # ndsb3manual_6a7f1fd0196a97568f2991c885ac1c0b_1_neg_0_3_1_pn.png
        parts = os.path.splitext(file_name)[0].split("_")
        if parts[0] == "ndsb3manual" or parts[0] == "hostpitalmanual":
            patient_id = parts[1]
            pn = parts[3]
        else:
            patient_id = parts[0]
            if parts[-1] == "luna" or parts[-1] == "edge": pn = "neg"
            else:  pn = parts[-1]

        return patient_id, pn

    @staticmethod
    def prepare_image_for_net3D(img):
        img = img.astype(numpy.float32)
        img -= DNN_model.MEAN_PIXEL_VALUE
        img /= 255.
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
        return img

    def filter_patient_nodules_predictions(self, df_nodule_predictions, patient_id, image_dir, view_size):
    # def filter_patient_nodules_predictions(self, df_nodule_predictions: pandas.DataFrame, patient_id, image_dir, view_size):
        patient_mask = helpers.load_patient_images(patient_id, image_dir, "*_m.png")
        delete_indices = []
        for index, row in df_nodule_predictions.iterrows():
            z_perc = row["coord_z"]
            y_perc = row["coord_y"]
            center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
            center_y = int(round(y_perc * patient_mask.shape[1]))
            center_z = int(round(z_perc * patient_mask.shape[0]))

            mal_score = row["diameter_mm"]
            start_y = center_y - view_size / 2
            start_x = center_x - view_size / 2
            nodule_in_mask = False
            for z_index in [-1, 0, 1]:
                img = patient_mask[z_index + center_z]
                start_x = int(start_x)
                start_y = int(start_y)
                view_size = int(view_size)
                img_roi = img[start_y:start_y + view_size, start_x:start_x + view_size]
                if img_roi.sum() > 255:  # more than 1 pixel of mask.
                    self.logger.info("More than 1 pixel of mask. nodule_in_mask is true")
                    nodule_in_mask = True

            if not nodule_in_mask:
                self.logger.info("Nodule not in mask: {0} {1} {2}".format(center_x, center_y, center_z))
                if mal_score > 0:
                    mal_score *= -1
                df_nodule_predictions.loc[index, "diameter_mm"] = mal_score
            else:
                if center_z < 30:
                    self.logger.info("Z < 30: {0} center z: {1}  y_perc: {2} ".format(patient_id, center_z, y_perc))
                    if mal_score > 0:
                        mal_score *= -1
                    df_nodule_predictions.loc[index, "diameter_mm"] = mal_score

                if (z_perc > 0.75 or z_perc < 0.25) and y_perc > 0.85:
                    self.logger.info("SUSPICIOUS FALSEPOSITIVE: {0}  center z: {1}  y_perc: {2}".format(patient_id, center_z,
                                                                                                   y_perc))

                if center_z < 50 and y_perc < 0.30:
                    self.logger.info(
                        "SUSPICIOUS FALSEPOSITIVE OUT OF RANGE: {0} center z: {1} y_perc: {2}".format(patient_id,
                                                                                                      center_z, y_perc))

        df_nodule_predictions.drop(df_nodule_predictions.index[delete_indices], inplace=True)
        return df_nodule_predictions

    def get_train_holdout_files(self, train_percentage=80, hospital_new_data=False, local_patient_set=False, full_luna_set=True, manual_labels=False, fold_count=0, ndsb3_holdout=0):
        self.logger.info("Get train/holdout files.")
        pos_samples_lidc = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_lidc/*.png")
        self.logger.info("Pos samples lidc: {0}".format(len(pos_samples_lidc)))

        pos_samples_manual = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_manual/*_pos.png")
        self.logger.info("Pos samples manual: {0}".format(len(pos_samples_manual)))
        pos_samples_original = pos_samples_lidc + pos_samples_manual
        self.logger.info("Original pos samples: {0}".format(len(pos_samples_original)))

        pos_samples_aug_lunalidc = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_lidc_aug/*.png")
        pos_samples_aug_lunamanual = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_manual_aug/*.png")
        # pos_samples_aug_ndsb3manual = glob.glob(settings.WORKING_DIR + "generated_traindata/ndsb3_train_cubes_manual_pos_aug/*.png")
        # pos_samples_aug = pos_samples_aug_lunalidc + pos_samples_aug_lunamanual + pos_samples_aug_ndsb3manual
        pos_samples_aug = pos_samples_aug_lunalidc + pos_samples_aug_lunamanual
        self.logger.info("Augmented pos samples: {0}".format(len(pos_samples_aug)))

        pos_samples = pos_samples_original + pos_samples_aug
        random.shuffle(pos_samples)

        train_pos_count = int((len(pos_samples) * train_percentage) / 100)
        pos_samples_train = pos_samples[:train_pos_count]
        pos_samples_holdout = pos_samples[train_pos_count:]

        if full_luna_set:      # True
            pos_samples_train += pos_samples_holdout
            if manual_labels:
                pos_samples_holdout = []

        ndsb3_list = glob.glob(settings.WORKING_DIR + "generated_traindata/ndsb3_train_cubes_manual/*.png")
        self.logger.info("Ndsb3 samples: {0} ".format(len(ndsb3_list)))

        pos_samples_ndsb3_fold = []
        pos_samples_ndsb3_holdout = []
        ndsb3_pos = 0
        ndsb3_neg = 0
        ndsb3_pos_holdout = 0
        ndsb3_neg_holdout = 0
        if manual_labels:      # False
            for file_path in ndsb3_list:
                file_name = ntpath.basename(file_path)
                parts = file_name.split("_")
                if int(parts[4]) == 0 and parts[3] != "neg":  # skip positive non-cancer-cases
                    continue

                if fold_count == 3:
                    if parts[3] == "neg":  # skip negative cases
                        continue

                patient_id = parts[1]
                patient_fold = helpers.get_patient_fold(patient_id) % fold_count
                if patient_fold == ndsb3_holdout:
                    self.logger.info("In holdout: {0}".format(patient_id))
                    pos_samples_ndsb3_holdout.append(file_path)
                    if parts[3] == "neg":
                        ndsb3_neg_holdout += 1
                    else:
                        ndsb3_pos_holdout += 1
                else:
                    pos_samples_ndsb3_fold.append(file_path)
                    self.logger.info("In fold: {0}".format(patient_id))
                    if parts[3] == "neg":
                        ndsb3_neg += 1
                    else:
                        ndsb3_pos += 1

        self.logger.info("{0} ndsb3 pos labels train".format(ndsb3_pos))
        self.logger.info("{0} ndsb3 neg labels train".format(ndsb3_neg))
        self.logger.info("{0} ndsb3 pos labels holdout".format(ndsb3_pos_holdout))
        self.logger.info("{0} ndsb3 neg labels holdout".format(ndsb3_neg_holdout))

        pos_samples_hospital_train = []
        pos_samples_hospital_holdout = []
        if local_patient_set:
            self.logger.info("Including hospital cases...")
            hospital_list = glob.glob(settings.WORKING_DIR + "generated_traindata/hospital_train_cubes_manual/*.png")
            random.shuffle(hospital_list)
            train_hospital_count = int((len(hospital_list) * train_percentage) / 100)
            pos_samples_hospital_train = hospital_list[:train_hospital_count]
            pos_samples_hospital_holdout = hospital_list[train_hospital_count:]

        if manual_labels:
            for times_ndsb3 in range(4):  # make ndsb labels count 4 times just like in LIDC when 4 doctors annotated a nodule
                pos_samples_train += pos_samples_ndsb3_fold
                pos_samples_holdout += pos_samples_ndsb3_holdout

        # 医生人工标注的新数据
        pos_samples_hospitalnew_train = []
        pos_samples_hospitalnew_holdout = []
        if hospital_new_data:
            self.logger.info("Including hospital new cases...")
            hospital_new_list = glob.glob(settings.WORKING_DIR + "generated_traindata/manualanno_train_cubes/*.png")
            random.shuffle(hospital_new_list)
            train_hospitalnew_count = int((len(hospital_new_list) * train_percentage) / 100)
            pos_samples_hospitalnew_train = hospital_new_list[:train_hospitalnew_count]
            pos_samples_hospitalnew_holdout = hospital_new_list[train_hospitalnew_count:]

        # 医生对模型预测结果进行修改的数据
        # ......

        '''neg samples'''
        neg_samples_edge = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_auto/*_edge.png")
        self.logger.info("Neg samples edge: {0}".format(len(neg_samples_edge)))

        # neg_samples_white = glob.glob(settings.BASE_DIR_SSD + "luna16_train_cubes_auto/*_white.png")
        neg_samples_luna = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_auto/*_luna.png")
        self.logger.info("Neg samples luna: {0}".format(len(neg_samples_luna)))

        # neg_samples = neg_samples_edge + neg_samples_white
        neg_samples = neg_samples_edge + neg_samples_luna
        random.shuffle(neg_samples)

        train_neg_count = int((len(neg_samples) * train_percentage) / 100)
        # train_neg_count = int((len(neg_samples) * 6) / 7)

        neg_samples_falsepos = []
        for file_path in glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_auto/*_falsepos.png"):
            neg_samples_falsepos.append(file_path)
        self.logger.info("Neg samples falsepos LUNA count: {0}".format(len(neg_samples_falsepos)))

        neg_samples_train = neg_samples[:train_neg_count]
        neg_samples_train += neg_samples_falsepos + neg_samples_falsepos + neg_samples_falsepos
        neg_samples_holdout = neg_samples[train_neg_count:]
        if full_luna_set:
            neg_samples_train += neg_samples_holdout

        self.logger.info("Train positive samples: {0}".format(len(pos_samples_train)))
        self.logger.info("Train negative samples: {0}".format(len(neg_samples_train)))
        self.logger.info("Train hospital samples: {0}".format(len(pos_samples_hospital_train)))
        self.logger.info("Train hospital new samples: {0}".format(len(pos_samples_hospitalnew_train)))

        self.logger.info("Holdout positive samples: {0}".format(len(pos_samples_holdout)))
        self.logger.info("Holdout negative samples: {0}".format(len(neg_samples_holdout)))
        self.logger.info("Holdout hospital samples: {0}".format(len(pos_samples_hospital_holdout)))
        self.logger.info("Holdout hospital new samples: {0}".format(len(pos_samples_hospitalnew_holdout)))

        train_res, holdout_res = [], []
        sets = [(train_res, pos_samples_train, neg_samples_train, pos_samples_hospital_train, pos_samples_hospitalnew_train),
                (holdout_res, pos_samples_holdout, neg_samples_holdout, pos_samples_hospital_holdout, pos_samples_hospitalnew_holdout)]

        for set_item in sets:
            pos_idx = 0
            negs_per_pos = self.NEGS_PER_POS
            res = set_item[0]
            pos_samples = set_item[1]
            neg_samples = set_item[2]
            hospital_samples = set_item[3]
            hospitalnew_samples = set_item[4]
            self.logger.info("Pos: {0}".format(len(pos_samples)))
            ndsb3_pos = 0
            ndsb3_neg = 0
            for index, neg_sample_path in enumerate(neg_samples):
                # res.append(sample_path + "/")
                res.append((neg_sample_path, 0, 0))
                # logger.info("index:{0}".format(index))
                if index % negs_per_pos == 0:
                    pos_sample_path = pos_samples[pos_idx]
                    file_name = ntpath.basename(pos_sample_path)
                    parts = file_name.split("_")

                    # logger.info("parts:{0}".format(parts))

                    if parts[0].startswith("ndsb3manual"):
                        if parts[3] == "pos":
                            class_label = 1  # only take positive examples where we know there was a cancer..
                            # cancer_label = int(parts[4])
                            # if cancer_label == 0:
                            #     print('ndsb3manual: positive example but no cancer: {0}'.format(parts))
                            #     continue
                            size_label = int(parts[5])
                            # logger.info(parts[1], size_label)
                            # assert class_label == 1
                            # logger.info("class_label:{0}".format(class_label))
                            if size_label < 1:
                                self.logger.info("{0} nodule size < 1".format(pos_sample_path))
                            assert size_label >= 1
                            ndsb3_pos += 1
                        else:
                            class_label = 0
                            size_label = 0
                            ndsb3_neg += 1
                    else:
                        # logger.info("parts:{0}".format(parts))
                        # logger.info("parts[3]: {0}".format(parts[3]))
                        class_label = int(parts[3])
                        # logger.info("class_label: {0}".format(class_label))
                        size_label = int(parts[2])
                        # logger.info("size_label: {0}".format(size_label))
                        assert class_label == 1
                        # logger.info("parts[4]:{0}".format(parts[4]))
                        # assert (parts[4] == "pos.png") or (parts[4] == "pos")
                        assert size_label >= 1
                        # logger.info("assert is done")

                    res.append((pos_sample_path, class_label, size_label))
                    # logger.info("res: {0}".format(res))

                    pos_idx += 1
                    pos_idx %= len(pos_samples)
                    # logger.info("one loop end.and pos_idx:{0}".format(pos_idx))
                    # ===================不重复取pos samples
                    # if pos_idx % len(pos_samples) == 0:
                    #    break

            if local_patient_set:
                for index, hospital_sample_path in enumerate(hospital_samples):
                    file_name = os.path.basename(hospital_sample_path)
                    parts = file_name.split("_")
                    if parts[3] == "pos":
                        class_label = 1
                    else:
                        class_label = 0
                    size_label = int(parts[5])
                    if size_label < 1:
                        self.logger.info("{0} nodule size < 1".format(file_name))
                    self.logger.info(
                        "Add sample {0} class: {1} size: {2}".format(hospital_sample_path, class_label, size_label))
                    res.append((hospital_sample_path, class_label, size_label))

            # 新标注数据的预处理方式和前面的local patient要一致（命名规则)
            if hospital_new_data:
                for index, hospital_sample_path in enumerate(hospitalnew_samples):
                    file_name = os.path.basename(hospital_sample_path)
                    parts = file_name.split("_")
                    if parts[3] == "pos":
                        class_label = 1
                    else:
                        class_label = 0
                    size_label = int(parts[5])
                    if size_label < 1:
                        self.logger.info("{0} nodule size < 1".format(file_name))
                    self.logger.info(
                        "Add sample {0} class: {1} size: {2}".format(hospital_sample_path, class_label, size_label))
                    res.append((hospital_sample_path, class_label, size_label))

            self.logger.info("ndsb3 pos: {0}".format(ndsb3_pos))
            self.logger.info("ndsb3 neg: {0}".format(ndsb3_neg))

        self.logger.info("Train count: {0}, holdout count: {1} ".format(len(train_res), len(holdout_res)))

        return train_res, holdout_res

    def data_generator(self, batch_size, record_list, train_set):
        batch_idx = 0
        means = []
        random_state = numpy.random.RandomState(1301)
        while True:
            img_list = []
            class_list = []
            size_list = []
            if train_set:
                random.shuffle(record_list)
            CROP_SIZE = self.CUBE_SIZE
            # CROP_SIZE = 48
            for record_idx, record_item in enumerate(record_list):
                # rint patient_dir
                class_label = record_item[1]
                size_label = record_item[2]
                if class_label == 0:
                    cube_image = helpers.load_cube_img(record_item[0], 6, 8, 48)
                    # if train_set:
                    #     # helpers.save_cube_img("c:/tmp/pre.png", cube_image, 8, 8)
                    #     cube_image = random_rotate_cube_img(cube_image, 0.99, -180, 180)
                    #
                    # if train_set:
                    #     if random.randint(0, 100) > 0.1:
                    #         # cube_image = numpy.flipud(cube_image)
                    #         cube_image = elastic_transform48(cube_image, 64, 8, random_state)
                    wiggle = 48 - CROP_SIZE - 1
                    indent_x = 0
                    indent_y = 0
                    indent_z = 0
                    if wiggle > 0:
                        indent_x = random.randint(0, wiggle)
                        indent_y = random.randint(0, wiggle)
                        indent_z = random.randint(0, wiggle)
                    cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE,
                                 indent_x:indent_x + CROP_SIZE]

                    if train_set:
                        if random.randint(0, 100) > 50:
                            cube_image = numpy.fliplr(cube_image)
                        if random.randint(0, 100) > 50:
                            cube_image = numpy.flipud(cube_image)
                        if random.randint(0, 100) > 50:
                            cube_image = cube_image[:, :, ::-1]
                        if random.randint(0, 100) > 50:
                            cube_image = cube_image[:, ::-1, :]

                    if CROP_SIZE != self.CUBE_SIZE:
                        cube_image = helpers.rescale_patient_images2(cube_image, (self.CUBE_SIZE, self.CUBE_SIZE, self.CUBE_SIZE))
                    assert cube_image.shape == (self.CUBE_SIZE, self.CUBE_SIZE, self.CUBE_SIZE)
                else:
                    cube_image = helpers.load_cube_img(record_item[0], 8, 8, 64)

                    if train_set:
                        pass

                    current_cube_size = cube_image.shape[0]
                    indent_x = (current_cube_size - CROP_SIZE) / 2
                    indent_y = (current_cube_size - CROP_SIZE) / 2
                    indent_z = (current_cube_size - CROP_SIZE) / 2
                    wiggle_indent = 0
                    wiggle = current_cube_size - CROP_SIZE - 1
                    if wiggle > (CROP_SIZE / 2):
                        wiggle_indent = CROP_SIZE / 4
                        wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1
                    if train_set:
                        indent_x = wiggle_indent + random.randint(0, wiggle)
                        indent_y = wiggle_indent + random.randint(0, wiggle)
                        indent_z = wiggle_indent + random.randint(0, wiggle)

                    indent_x = int(indent_x)
                    indent_y = int(indent_y)
                    indent_z = int(indent_z)
                    cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE,
                                 indent_x:indent_x + CROP_SIZE]
                    if CROP_SIZE != self.CUBE_SIZE:
                        cube_image = helpers.rescale_patient_images2(cube_image, (self.CUBE_SIZE, self.CUBE_SIZE, self.CUBE_SIZE))
                    assert cube_image.shape == (self.CUBE_SIZE, self.CUBE_SIZE, self.CUBE_SIZE)

                    if train_set:
                        if random.randint(0, 100) > 50:
                            cube_image = numpy.fliplr(cube_image)
                        if random.randint(0, 100) > 50:
                            cube_image = numpy.flipud(cube_image)
                        if random.randint(0, 100) > 50:
                            cube_image = cube_image[:, :, ::-1]
                        if random.randint(0, 100) > 50:
                            cube_image = cube_image[:, ::-1, :]

                means.append(cube_image.mean())
                img3d = self.prepare_image_for_net3D(cube_image)
                if train_set:
                    if len(means) % 1000000 == 0:
                        self.logger.info("Mean: {0}".format(sum(means) / len(means)))
                img_list.append(img3d)
                class_list.append(class_label)
                size_list.append(size_label)

                batch_idx += 1
                if batch_idx >= batch_size:
                    x = numpy.vstack(img_list)
                    y_class = numpy.vstack(class_list)
                    y_size = numpy.vstack(size_list)
                    yield x, {"out_class": y_class, "out_malignancy": y_size}
                    img_list = []
                    class_list = []
                    size_list = []
                    batch_idx = 0

    def train(self, model_name, train_epoch_save_folder='', train_model_save_folder='', epoch_number=12, batch_size=16):
        if self.model is None:
            self.logger.error("The model is None. No training happens.")
            return

        if self.trained:
            return

        train_files, holdout_files = self.get_train_holdout_files(train_percentage=80, hospital_new_data=True, local_patient_set=False, full_luna_set=True, manual_labels=False)
        self.logger.info("get_train_holdout_files is done.")
        # train_files = train_files[:100]
        # holdout_files = train_files[:10]
        train_gen = self.data_generator(batch_size, train_files, train_set=True)
        holdout_gen = self.data_generator(batch_size, holdout_files, train_set=False)

        for i in range(0, 10):
            tmp = next(holdout_gen)
            cube_img = tmp[0][0].reshape(self.CUBE_SIZE, self.CUBE_SIZE, self.CUBE_SIZE, 1)
            cube_img = cube_img[:, :, :, 0]
            cube_img *= 255.
            cube_img += self.MEAN_PIXEL_VALUE
            # helpers.save_cube_img("c:/tmp/img_" + str(i) + ".png", cube_img, 4, 8)
            # logger.info(tmp)

        logcallback = LoggingCallback(self.logger.info)
        self.callbacks.append(logcallback)

        learnrate_scheduler = LearningRateScheduler(self.step_decay)
        self.callbacks.append(learnrate_scheduler)

        if self.tensorboard_enabled:
            if not os.path.exists(self.TENSORBOARD_LOG_DIR):
                os.makedirs(self.TENSORBOARD_LOG_DIR)

            tensorboard_callback = TensorBoard(
                log_dir=self.TENSORBOARD_LOG_DIR,
                histogram_freq=2,
                # write_images=True, # Enabling this line would require more than 5 GB at each `histogram_freq` epoch.
                write_graph=True
                # embeddings_freq=3,
                # embeddings_layer_names=list(embeddings_metadata.keys()),
                # embeddings_metadata=embeddings_metadata
            )
            tensorboard_callback.set_model(self.model)
            self.callbacks.append(tensorboard_callback)

        checkpoint_epoch_model = ModelCheckpoint(
            train_epoch_save_folder + "model_" + model_name + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5",
            monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto',
            period=1)
        checkpoint_best_model = ModelCheckpoint(
            train_epoch_save_folder + "model_" + model_name + "_best.hd5", monitor='val_loss',
            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.callbacks.append(checkpoint_epoch_model)
        self.callbacks.append(checkpoint_best_model)

        train_history = self.model.fit_generator(train_gen, len(train_gen) / batch_size, epoch_number,
                                                 validation_data=holdout_gen,
                                                 validation_steps=len(holdout_gen) / batch_size,
                                                 callbacks=self.callbacks)
        self.logger.info("Model fit_generator finished.")
        self.model.save(train_epoch_save_folder + "model_" + model_name + "_end.hd5")

        pandas.DataFrame(train_history.history).to_csv(train_epoch_save_folder + "model_" + model_name + "_history.csv")
        self.trained = True

        shutil.copy(train_epoch_save_folder + "model_" + model_name + "_best.hd5", train_model_save_folder + "model_" + model_name + "_best.hd5")

    def predict_imagelist(self, img_list):
        if self.model is None:
            self.logger.error("The model is None. Please call generate_model() to generate the model at first.")
            return None
        batch_size = 1  # for test
        batch_list = []
        batch_list_loc = []
        count = 0
        predictions = []

        for item in img_list:
            cube_img = item[0]
            file_name = item[1]
            patient_id = self.analysis_filename(file_name)[0]
            self.logger.info("====={0} - patient_id {1}".format(count, patient_id))
            # logger.info("the shape of cube image: {0}".format(numpy.array(cube_img).shape)) # (1, 32, 32, 32, 1)
            count += 1
            batch_list.append(cube_img)
            batch_list_loc.append(file_name)
            # logger.info("batch list: {0}".format(batch_list))
            # logger.info("the shape of batch list: {0}".format(numpy.array(batch_list).shape)) # (1, 1, 32, 32, 32, 1)
            # logger.info("batch list loc: {0}".format(batch_list_loc))

            # if len(batch_list) % batch_size == 0:
            batch_data = numpy.vstack(batch_list)
            p = self.model.predict(batch_data, batch_size=batch_size)
            self.logger.info("the prediction result p: {0}".format(p))
            # [array([[ 0.00064842]], dtype=float32), array([[  1.68593288e-05]], dtype=float32)]
            self.logger.info("the shape of p:{0}".format(numpy.array(p).shape))  # (2, 1, 1)
            self.logger.info("the length of p[0]:{0}".format(len(p[0])))  # 1

            # for i in range(len(p[0])):
            i = 0
            file_name = batch_list_loc[i]
            nodule_chance = p[0][i][0]
            diameter_mm = round(p[1][i][0], 4)
            nodule_chance = round(nodule_chance, 4)
            self.logger.info("nodule chance:{0}, diameter_mm:{1}".format(nodule_chance, diameter_mm))
            item_prediction = [file_name, nodule_chance, diameter_mm]
            predictions.append(item_prediction)

            batch_list = []
            batch_list_loc = []
            # count = 0

        return predictions

    def predict_patient(self, patient_id, image_dir, result_dir, magnification=1, flip=False):
        if self.model is None:
            self.logger.error("The model is None. Please call generate_model() to generate the model at first.")
            return None
        
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        sw = helpers.Stopwatch.start_new()
        csv_target_path = result_dir + patient_id + ".csv"
        all_predictions_csv = []
        patient_img = helpers.load_patient_images(patient_id, image_dir, "*_i.png", [])
        if magnification != 1:
            patient_img = helpers.rescale_patient_images(patient_img, (1, 1, 1), magnification)

        patient_mask = helpers.load_patient_images(patient_id, image_dir, "*_m.png", [])
        if magnification != 1:
            patient_mask = helpers.rescale_patient_images(patient_mask, (1, 1, 1), magnification, is_mask_image=True)

        step = self.PREDICT_STEP
        CROP_SIZE = self.CUBE_SIZE
        # CROP_SIZE = 48

        predict_volume_shape_list = [0, 0, 0]
        for dim in range(3):
            dim_indent = 0
            while dim_indent + CROP_SIZE < patient_img.shape[dim]:
                predict_volume_shape_list[dim] += 1
                dim_indent += step

        predict_volume_shape = (
        predict_volume_shape_list[0], predict_volume_shape_list[1], predict_volume_shape_list[2])
        predict_volume = numpy.zeros(shape=predict_volume_shape, dtype=float)
        self.logger.info("Predict volume shape: {0}".format(predict_volume.shape))
        done_count = 0
        skipped_count = 0
        batch_size = 128
        batch_list = []
        batch_list_coords = []
        patient_predictions_csv = []
        cube_img = None
        annotation_index = 0

        for z in range(0, predict_volume_shape[0]):
            for y in range(0, predict_volume_shape[1]):
                for x in range(0, predict_volume_shape[2]):
                    # if cube_img is None:
                    cube_img = patient_img[z * step:z * step + CROP_SIZE, y * step:y * step + CROP_SIZE,
                               x * step:x * step + CROP_SIZE]
                    cube_mask = patient_mask[z * step:z * step + CROP_SIZE, y * step:y * step + CROP_SIZE,
                                x * step:x * step + CROP_SIZE]

                    if cube_mask.sum() < 2000:
                        skipped_count += 1
                        self.logger.info("Cube x {0} y  {1} z {2} is skipped!!!".format(x, y, z))
                    else:
                        if flip:
                            cube_img = cube_img[:, :, ::-1]

                        if CROP_SIZE != self.CUBE_SIZE:
                            cube_img = helpers.rescale_patient_images2(cube_img, (self.CUBE_SIZE, self.CUBE_SIZE, self.CUBE_SIZE))
                            # helpers.save_cube_img("c:/tmp/cube.png", cube_img, 8, 4)
                            # cube_mask = helpers.rescale_patient_images2(cube_mask, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))

                        img_prep = self.prepare_image_for_net3D(cube_img)
                        batch_list.append(img_prep)
                        batch_list_coords.append((z, y, x))
                        if len(batch_list) % batch_size == 0:
                            batch_data = numpy.vstack(batch_list)
                            p = self.model.predict(batch_data, batch_size=batch_size)
                            for i in range(len(p[0])):
                                p_z = batch_list_coords[i][0]
                                p_y = batch_list_coords[i][1]
                                p_x = batch_list_coords[i][2]
                                nodule_chance = p[0][i][0]
                                predict_volume[p_z, p_y, p_x] = nodule_chance
                                if nodule_chance > self.P_TH:
                                    self.logger.info(
                                        "Cube x {0} y  {1} z {2} is possible nodule, nodule_chance is {3}!!!".format(
                                            p_x, p_y, p_z, nodule_chance))
                                    p_z = p_z * step + CROP_SIZE / 2
                                    p_y = p_y * step + CROP_SIZE / 2
                                    p_x = p_x * step + CROP_SIZE / 2

                                    self.logger.info("Cube center x {0} y {1} z {2} ".format(p_x, p_y, p_z))
                                    p_z_perc = round(p_z / patient_img.shape[0], 4)
                                    p_y_perc = round(p_y / patient_img.shape[1], 4)
                                    p_x_perc = round(p_x / patient_img.shape[2], 4)
                                    diameter_mm = round(p[1][i][0], 4)
                                    self.logger.info("Cube diameter_mm {0} ".format(diameter_mm))
                                    # diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    diameter_perc = round(diameter_mm / patient_img.shape[2], 4)
                                    self.logger.info(
                                        "Cube center percentage x {0} y {1} z {2} diamm {3} ".format(p_x_perc, p_y_perc,
                                                                                                     p_z_perc,
                                                                                                     diameter_mm))
                                    nodule_chance = round(nodule_chance, 4)
                                    patient_predictions_csv_line = [annotation_index, p_x_perc, p_y_perc, p_z_perc,
                                                                    diameter_perc, nodule_chance, diameter_mm]
                                    patient_predictions_csv.append(patient_predictions_csv_line)
                                    all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                    annotation_index += 1

                            batch_list = []
                            batch_list_coords = []
                    done_count += 1
                    if done_count % 10000 == 0:
                        self.logger.info("Done: {0} skipped: {1}".format(done_count, skipped_count))

        df = pandas.DataFrame(patient_predictions_csv,
                              columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter",
                              "nodule_chance", "diameter_mm"])

        self.filter_patient_nodules_predictions(df, patient_id, image_dir, CROP_SIZE * magnification)
        df.to_csv(csv_target_path, index=False)
        self.logger.info("predict_volume mean is {0}".format(predict_volume.mean()))
        self.logger.info("Done in {0} seconds".format(sw.get_elapsed_seconds()))
