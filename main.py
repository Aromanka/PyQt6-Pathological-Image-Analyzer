from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLineEdit,
                             QLabel, QTreeView, QStatusBar, QFileDialog, QCheckBox,
                             QListWidget, QListWidgetItem, QToolBar, QColorDialog,
                             QSpinBox, QVBoxLayout, QFrame, QSizePolicy, QWidget)
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction, QImage, QPixmap, QIcon
from PyQt6 import uic
from PyQt6.QtCore import Qt, QSize, QRect, QEvent, QUrl
# from PyQt6.QtWebEngineWidgets import QWebEngineView
import sys
import cv2
import os
import numpy as np
from LayerManagement import LayerManagement
import ProcessingFunctions as PF
from MyQWidgets import LayerCheckBox, AboutDialog, TutorialDialog, StreamThread
from report.generate_report import create_report, PDF_PATH
import qdarkstyle
import fitz
from utils import copy_and_rename_file


# ------------Guidelines-----------
# 1. All images are processed in cv2.Mat form. And all 3-channels images are processed in COLOR_BGR order.
# ---------------------------------
# bug_to_fix:
# todo: Foreground Segmentation rename -> Mask Segmentation
# todo: draw indicating ellipses around ROI regions


def get_treeView_operations_model():
    model = QStandardItemModel()
    model.setHorizontalHeaderLabels(["Operation"])

    class1 = QStandardItem("基础操作")
    class1.appendRow([QStandardItem("Grayscale")])
    class1.appendRow([QStandardItem("Binary")])
    class1.appendRow([QStandardItem("Color Correction")])
    class1.appendRow([QStandardItem("Mean Filter")])
    class1.appendRow([QStandardItem("Median Filter")])
    class1.appendRow([QStandardItem("Gaussian Filter")])
    class1.appendRow([QStandardItem("Foreground Mask")])
    class1.appendRow([QStandardItem("ROI Mask")])
    class1.appendRow([QStandardItem("Draw Contour")])
    class1.appendRow([QStandardItem("Draw ROI Contour")])
    model.appendRow(class1)

    class2 = QStandardItem("复合操作")
    # class2.appendRow([QStandardItem("Denoise")])
    class2.appendRow([QStandardItem("Foreground Contour")])
    class2.appendRow([QStandardItem("Foreground Segmentation")])
    class2.appendRow([QStandardItem("Foreground Segmentation & Save ROI")])
    model.appendRow(class2)

    class3 = QStandardItem("快捷操作")
    # class2.appendRow([QStandardItem("Denoise")])
    class3.appendRow([QStandardItem("Locate ROI")])
    class3.appendRow([QStandardItem("Analyze Portal Regions")])
    model.appendRow(class3)

    return model


def cv2_2_QImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    return QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()


class DemoWindow(QMainWindow):
    auto_save_dir = r"../data/export_images/"
    # toolBar_height = 30
    LABEL_IMAGE_SCALE_MIN = 1
    LABEL_IMAGE_SCALE_MAX = 20
    DEFAULT_SCALING = 0.1

    PORTAL_REGION_COLOR = (0, 255, 0) # GREEN
    NORMAL_PORTAL_REGION_COLOR = (0, 0, 0)  # BLACK
    FIBROSIS_PORTAL_REGION_COLOR = (255, 0, 0)  # RED

    def __init__(self, parent=None):
        super(DemoWindow, self).__init__(parent)
        self.report_info = {
            "patient_name": '',
            "patient_age": '',
            "patient_gender": '',
            "patient_id": '',
            "image_name": '',
            "image_resolution": '',
            "portal_num": '',
            "normal_portal_num": '',
            "fibre_portal_num": '',
            "portal_area": '',
            "normal_portal_area": '',
            "fibre_portal_area": ''
        }

        self.ui = uic.loadUi("./window.ui", self)
        self.initUI()
        self.img = None
        self.img_size = None
        self.img_scaling = self.DEFAULT_SCALING
        self.scaled_img = None
        self.filename = None
        self.img_filepath = None
        self.cur_select_imgname = None
        self.cur_select_maskname = None
        self.label_pixmap = None
        self.spinBoxesSync = False
        self.cur_color = (0,0,0)
        self.layer_management = LayerManagement()

        # mouse control
        self.angel = None
        self.angleY = None
        self.label_image_scaling = 10   # int: [1:20]
        self.cur_img = None
        self.cur_pixmap = None
        self.label_image_x1 = self.label_image_y1 = ""
        self.label_image_offsetX = self.label_image_offsetY = ""
        self.mousePressing = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # clear tool box
        self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':0})

        # OpenAI API settings
        self.stream_thread = None


    def initUI(self):
        # self.treeView_operations : QTreeView = self.ui.treeView_operations
        self.treeView_operations.setModel(get_treeView_operations_model())
        self.treeView_operations.expandAll()
        self.treeView_operations.selectionModel().currentChanged.connect(self.on_treeView_operations_select_changed)
        # self.treeView_operations.setFixedWidth(300)
        # self.treeView_operations.resize(200, self.treeView_operations.height())

        self.statusbar_label = QLabel()
        self.statusbar_label.setText("status bar")
        self.statusbar.addWidget(self.statusbar_label)

        self.label_image = QLabel(self.frame)
        self.label_image.setFrameShape(QFrame.Shape.Box)
        self.label_image.setLineWidth(1)
        self.label_x = self.label_image.x()
        self.label_y = self.label_image.y()
        canvas_size_scale = 0.8
        self.label_w = int(self.frame.width() * canvas_size_scale)
        self.label_h = int(self.frame.height() * canvas_size_scale)
        # self.label_image.setScaledContents(True)

        # self.label_info : QLabel = self.ui.label_info
        # self.label_info.setText("")

        self.action_operate.triggered.connect(self.on_operate)
        self.action_openFile.triggered.connect(self.on_openFile)
        self.action_ROISaveAs.triggered.connect(self.on_ROISaveAs)
        self.action_CanvasSaveAs.triggered.connect(self.on_CanvasSaveAs)
        self.action_new.triggered.connect(self.on_new)
        self.action_new.triggered.connect(self.on_new)
        self.action_savePDF.triggered.connect(self.on_PDFSaveAs)
        self.action_exit.triggered.connect(self.close)
        self.action_aboutSoftware.triggered.connect(self.show_about_dialog)
        self.action_tutorial.triggered.connect(self.show_tutorial_dialog)

        img = QImage(r'./ico/play_triangle.png')
        pixmap = QPixmap(img)
        ico_w, ico_h = 24, 24
        pixmap = pixmap.scaled(ico_w, ico_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        ico = QIcon(pixmap)
        self.pushBtn_actionOperate.setIcon(ico)
        self.pushBtn_actionOperate.setIconSize(QSize(ico_w, ico_h))
        self.pushBtn_actionOperate.setFixedSize(QSize(ico_w, ico_h))
        self.pushBtn_actionOperate.setText("")
        self.pushBtn_actionOperate.clicked.connect(self.on_operate)

        # self.label_ksize = self.ui.label_ksize
        # self.spinBox_ksizeW : QSpinBox = self.ui.spinBox_ksizeW
        self.spinBox_ksizeW.setMinimum(1)
        self.spinBox_ksizeW.setMaximum(100)
        # self.spinBox_ksizeH : QSpinBox = self.ui.spinBox_ksizeH
        self.spinBox_ksizeH.setMinimum(1)
        self.spinBox_ksizeH.setMaximum(100)
        self.spinBox_ksizeW.valueChanged.connect(self.on_spinBoxW_value_changed)

        # self.pushBtn_chooseColor : QPushButton = self.ui.pushButton_chooseColor
        self.pushBtn_chooseColor.clicked.connect(self.on_chooseColor)

        self.spinBox_scale.setMinimum(0.01)
        self.spinBox_scale.setMaximum(1)
        self.spinBox_scale.setValue(self.DEFAULT_SCALING)
        self.spinBox_scale.valueChanged.connect(self.on_rescaling)

        # set tabWidgets
        self.tabWidget.currentChanged.connect(self.on_PDFReport)
        self.tabWidget.setCurrentIndex(0)
        self.reload_report()

        self.pushBtn_savePDF.clicked.connect(self.on_PDFSaveAs)

        # set patient info interfaces
        self.spinBox_patientAge.setMinimum(0)
        self.spinBox_patientAge.setMaximum(130)

        # set AI frame
        self.pushBtn_AIGenerate.clicked.connect(self.start_stream)
        self.pushBtn_AIStop.clicked.connect(self.stop_stream)
        self.pushBtn_AIStop.setEnabled(False)


    def get_default_labelwh(self, img_size):
        print(f'ims:{img_size}')
        canvas_size_scale = 0.8
        w, h = img_size
        self.label_w = int(self.frame.width() * canvas_size_scale)
        self.label_h = int(self.frame.height() * canvas_size_scale)
        # print(f" img_wh: {w}*{h}")
        # print(f" max-label_wh: {self.label_w}*{self.label_h}")
        scaling_ratio_w = self.label_w / w
        scaling_ratio_h = self.label_h / h
        # self.label_image_scaling = np.clip(int(min(scaling_ratio_w, scaling_ratio_h)),
        #                                    self.LABEL_IMAGE_SCALE_MIN, self.LABEL_IMAGE_SCALE_MAX)
        scaling = np.clip(int(min(scaling_ratio_w, scaling_ratio_h)),
                                           self.LABEL_IMAGE_SCALE_MIN, self.LABEL_IMAGE_SCALE_MAX)
        # if scaling_ratio_w < scaling_ratio_h:
        #     self.label_h = int(h * scaling_ratio_w)
        # else:
        #     self.label_w = int(w * scaling_ratio_h)
        # print(f" label_wh: {self.label_w}*{self.label_h}")
        self.label_w = w * scaling
        self.label_h = h * scaling
        return self.label_w, self.label_h


    def on_spinBoxW_value_changed(self):
        if self.spinBoxesSync:
            self.spinBox_ksizeH.setValue(self.spinBox_ksizeW.value())

    def on_treeView_operations_select_changed(self, current : QStandardItem, previous):
        cur_operation = str(current.data())
        txt = f"class: {str(current.parent().data())}, operation: {cur_operation}"
        self.statusbar.showMessage(txt)
        if cur_operation == 'Grayscale':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':0})
        elif cur_operation == 'Binary':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':0})
        elif cur_operation == 'Mean Filter':
            self.set_toolBox({'spinBox_ksizeW':1, 'spinBox_ksizeH':1, 'choose_color':0, 'remove_small':0})
        elif cur_operation == 'Color Correction':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':0})
        elif cur_operation == 'Gaussian Filter':
            # todo: tool box sigma
            self.set_toolBox({'spinBox_ksizeW':1, 'spinBox_ksizeH':1, 'choose_color':0, 'remove_small':0})
        elif cur_operation == 'Median Filter':
            self.set_toolBox({'spinBox_ksizeW':1, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':0})
        elif cur_operation == 'Draw Contour':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':1, 'remove_small':0})
        elif cur_operation == 'Draw ROI Contour':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':1, 'remove_small':0})
        elif cur_operation == 'Foreground Contour':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':1, 'remove_small':0})
        elif cur_operation == 'Foreground Mask':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':0})
        elif cur_operation == 'ROI Mask':
            self.set_toolBox({'spinBox_ksizeW':1, 'spinBox_ksizeH':1, 'choose_color':0, 'remove_small':0})
        elif cur_operation == 'Foreground Segmentation':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':1})
        elif cur_operation == 'Foreground Segmentation & Save ROI':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':1})
        elif cur_operation == 'Locate ROI':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':1, 'remove_small':0})
        elif cur_operation == 'Analyze Portal Regions':
            self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':0})

    # available operations:
    # Grayscale: obtain a grayscale img layer
    # Binary: obtain a binary img layer
    # Foreground Contour: obtain a foreground contour layer
    # Mean Filter: obtain a mean filtered img layer
    # Gaussian Filter: obtain a Gaussian filtered img layer
    def on_operate(self):
        # judge validity
        cur_operation = self.current_selected_operation()
        if cur_operation is None:
            txt = "No operation selected."
            print(txt)
            self.statusbar.showMessage(txt)
            return
        if self.scaled_img is None:
            txt = "No image to process!"
            self.statusbar.showMessage(txt)
            return
        txt = f"operate: {cur_operation}"
        self.statusbar.showMessage(txt)
        print(txt)

        # operate
        if self.cur_select_imgname is None:
            cur_img = self.scaled_img
        else:
            cur_img, _ = self.layer_management.get_layer_by_name(self.cur_select_imgname)
        if self.cur_select_maskname is None:
            cur_mask = None
        else:
            cur_mask, _ = self.layer_management.get_layer_by_name(self.cur_select_maskname)
        try:
            if cur_operation == 'Grayscale':
                img = PF.process_grayscale(cur_img)
                self.add_layer('Grayscale', img)
            elif cur_operation == 'Binary':
                img = PF.process_binary(cur_img)
                self.add_layer('Binary', img)
            elif cur_operation == 'Mean Filter':
                # params = {'ksize':(5,5)}
                params = {'ksize': self.get_spinBox_ksize()}
                img = PF.process(cur_img, PF.MEAN_FILTER ,params)
                self.add_layer('Mean Filter', img)
            elif cur_operation == 'Gaussian Filter':
                # params = {'ksize':(5,5), 'sigma':1.0}
                params = {'ksize': self.get_spinBox_ksize(), 'sigma':1.0}
                print(params)
                img = PF.process(cur_img, PF.GAUSSIAN_FILTER, params)
                self.add_layer('Gaussian Filter', img)
            elif cur_operation == 'Median Filter':
                # params = {'ksize':(5,)}
                params = {'ksize': self.get_spinBox_ksize()}
                img = PF.process(cur_img, PF.MEDIAN_FILTER, params)
                self.add_layer('Median Filter', img)
            elif cur_operation == 'Color Correction':
                if cur_mask is not None:
                    img = PF.color_correction(cur_img, mask=cur_mask)
                else:
                    img = PF.color_correction(cur_img)
                self.add_layer('Color Correction', img)
            elif cur_operation == 'Draw Contour':
                if cur_mask is not None:
                    self.add_layer('Contour', (cur_mask, self.cur_color, 'Contour'), is_contour=True)
                else:
                    print('No mask.')
            elif cur_operation == 'Draw ROI Contour':
                if cur_mask is not None:
                    self.add_layer('Convex Contour',
                                   (PF.smooth_expand_mask(cur_mask), self.cur_color, 'Contour'),
                                   is_contour=True)
                else:
                    print('No mask.')
            elif cur_operation == 'Foreground Contour':
                mask = PF.obtain_foreground_mask(cur_img, PF.FOREGROUND_HSV)
                self.add_layer('Foreground Contour', (mask, self.cur_color, 'Contour'), is_contour=True)
            elif cur_operation == 'Foreground Mask':
                mask = PF.obtain_foreground_mask(cur_img, PF.FOREGROUND_HSV)
                self.add_layer('Foreground Mask', (mask, self.cur_color, 'Mask'))
            elif cur_operation == 'ROI Mask':
                mask = PF.obtain_ROI_mask(cur_img, ksize_ratio=1/300, remove_noise_flag=True)
                # mask = PF.obtain_foreground_mask(cur_img, PF.ROI_HSV)
                self.add_layer('ROI Mask', (mask, self.cur_color, 'Mask'))
            elif cur_operation == 'Foreground Segmentation':
                remove_small = self.checkBox_tool.isChecked()
                if cur_mask is not None:       # cur_img is a binary mask
                    masks = PF.segment_regions(cur_mask, remove_small)
                else:                           # cur_img is a BGR
                    mask = PF.obtain_foreground_mask(cur_img, PF.FOREGROUND_HSV)
                    masks = PF.segment_regions(mask, remove_small)
                for _mask in masks:
                    self.add_layer('Foreground Mask', (_mask, self.cur_color, 'Mask'))
            elif cur_operation == 'Foreground Segmentation & Save ROI':
                remove_small = self.checkBox_tool.isChecked()
                # obtain segmented masks
                if len(cur_img.shape)==2:       # cur_img is a binary mask
                    masks = PF.segment_regions(cur_img, remove_small)
                else:                           # cur_img is a BGR img
                    _mask = PF.obtain_foreground_mask(cur_img, PF.FOREGROUND_HSV)
                    masks = PF.segment_regions(_mask, remove_small)
                for i, mask in enumerate(masks):
                    # self.add_layer('Foreground Mask', (mask, (255, 0, 0), 'Mask'))
                    # obtain & save ROI
                    img = self.layer_management.get_active_img()
                    if img is None:
                        img = self.scaled_img
                    img_name, _ = os.path.splitext(self.filename)
                    img_name = f'{img_name}_{i}'
                    save_path = get_save_path(self.auto_save_dir, img_name)
                    print(f'ROI_{i} saved to {save_path}.')
                    PF.save_ROI(img, mask, save_path)
            elif cur_operation == 'Locate ROI':
                mask = PF.obtain_ROI_mask(cur_img, ksize_ratio=1 / 300, remove_noise_flag=True)
                self.add_layer('ROI Contour',
                               (PF.smooth_expand_mask(mask), self.cur_color, 'Contour'),
                               is_contour=True)
            elif cur_operation == 'Analyze Portal Regions':
                # set image to full scale to analyze details
                self.spinBox_scale.setValue(1)
                self.on_rescaling()
                # reload img
                if self.cur_select_imgname is None:
                    cur_img = self.scaled_img
                else:
                    cur_img, _ = self.layer_management.get_layer_by_name(self.cur_select_imgname)
                # process
                portal_mask, normal_mask, fibre_mask, info = PF.detect_portal_areas_texture(cur_img)
                self.add_layer('Portal Regions',
                               (PF.smooth_expand_mask(portal_mask), self.PORTAL_REGION_COLOR, 'Contour'),
                               is_contour=True)
                self.add_layer('Normal Portal Regions',
                               (PF.smooth_expand_mask(normal_mask), self.NORMAL_PORTAL_REGION_COLOR, 'Contour'),
                               is_contour=True)
                self.add_layer('Fibrosis Portal Regions',
                               (PF.smooth_expand_mask(fibre_mask), self.FIBROSIS_PORTAL_REGION_COLOR, 'Contour'),
                               is_contour=True)
                for label, data in info.items():
                    self.report_info[label] = data

        except :
            print(f"Invalid selected image (shape: {cur_img.shape}).")

    # ------------------------ToolBoxSetting-------------------------

    def set_toolBox(self, act_widgets):
        """
        setting up the toolbox according to act_widgets

        :param act_widgets: dict of widgets active status
        :return: None
        """
        if act_widgets['spinBox_ksizeW'] & act_widgets['spinBox_ksizeH']:
            self.label_ksize.setVisible(True)
            self.spinBox_ksizeW.setVisible(True)
            self.spinBox_ksizeH.setVisible(True)
            self.spinBox_ksizeH.setReadOnly(False)
            self.spinBoxesSync = False
        elif act_widgets['spinBox_ksizeW']:
            self.label_ksize.setVisible(True)
            self.spinBox_ksizeW.setVisible(True)
            self.spinBox_ksizeH.setVisible(True)
            self.spinBox_ksizeH.setReadOnly(True)
            self.spinBoxesSync = True
        else:
            self.label_ksize.setVisible(False)
            self.spinBox_ksizeH.setReadOnly(False)
            self.spinBox_ksizeW.setVisible(False)
            self.spinBox_ksizeH.setVisible(False)
            # self.spinBoxesSync = False
        if act_widgets['choose_color']:
            self.pushBtn_chooseColor.setVisible(True)
            self.label_color.setVisible(True)
        else:
            self.pushBtn_chooseColor.setVisible(False)
            self.label_color.setVisible(False)
        if act_widgets['remove_small']:
            self.checkBox_tool.setVisible(True)
            self.checkBox_tool.setText("去除小片段")
        else:
            self.checkBox_tool.setVisible(False)


    # --------------------------OnAction-----------------------------

    def on_openFile(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                 "Open Name",
                                                 r"../data/",
                                                 "Image Files (*.png *.jpg *.bmp)")
        print(filename)
        if filename:
            self.on_new()
            print('reached')

            # view image
            self.img_filepath = filename
            self.filename = os.path.basename(filename)
            print(f"self.img_filepath updated: {self.img_filepath}")
            self.reload_img()
            self.on_metadata_changed()

    def on_metadata_changed(self):
        """
        Metadate is the information about image that cannot be directly modified by users.
        """
        width, height = self.img.shape[:-1]
        img_name, _ = os.path.splitext(self.filename)
        self.lineEdit_imageName.setText(img_name)
        self.lineEdit_originalResolution.setText(f"{height} * {width}")
        if self.cur_select_imgname is not None:
            self.lineEdit_selectedLayer.setText(self.cur_select_imgname)
        else:
            self.lineEdit_selectedLayer.setText("无")
        if self.cur_select_maskname is not None:
            self.lineEdit_selectedMask.setText(self.cur_select_maskname)
        else:
            self.lineEdit_selectedMask.setText("无")
        # self.label_info.setText(f"Image Name: {img_name}\n"
        #                         f"Original Resolution:\n"
        #                         f"{height} * {width}\n"
        #                         f"Scale: {self.img_scaling}\n"
        #                         f"Working Resolution:\n"
        #                         f"{self.scaled_img.shape[0]} * {self.scaled_img.shape[1]}\n"
        #                         f"current selected image:\n"
        #                         f"{self.cur_select_imgname}\n"
        #                         f"current selected mask:\n"
        #                         f"{self.cur_select_maskname}")

        # set report info
        self.report_info['image_name'] = img_name

    def on_ROISaveAs(self):
        if self.cur_select_imgname is not None and self.cur_select_maskname is not None:
            cur_img, _ = self.layer_management.get_layer_by_name(self.cur_select_imgname)
            cur_mask, _ = self.layer_management.get_layer_by_name(self.cur_select_maskname)
            filename, _ = QFileDialog.getSaveFileName(self, "Save Name",
                                                      r"../data/export_images/", "Image Files (*.png)")
            print(filename)
            if filename:
                PF.save_ROI(cur_img, cur_mask, filename)
        elif self.cur_select_imgname is None:
            print(f'Invalid selected Image: {self.cur_select_imgname}')
        elif self.cur_select_maskname is None:
            print(f'Invalid selected Mask: {self.cur_select_maskname}')

    def on_CanvasSaveAs(self):
        filename, filetype = QFileDialog.getSaveFileName(self,
                                                         "Save Name",
                                                         r"../data/export_images/",
                                                         "Image Files (*.png)")
        print(filename)
        if filename:
            img = self.layer_management.get_mixed_img()
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def on_PDFSaveAs(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Name",
                                                  r"../data/export_images/", "PDF (*.pdf)")
        basename = os.path.basename(filename)
        dirname = os.path.dirname(filename)
        pdfpath = r'./report/report.pdf'
        copy_and_rename_file(pdfpath, dirname, basename)
        txt = '{}已保存!'.format(filename)
        print(txt)
        self.statusbar.showMessage(txt)



    def on_new(self):   # reset
        # self.initUI()
        self.img = None
        self.img_size = None
        self.img_scaling = self.DEFAULT_SCALING
        self.scaled_img = None
        self.filename = None
        self.img_filepath = None
        self.cur_select_imgname = None
        self.cur_select_maskname = None
        self.listWidget_layer.clear()
        self.layer_management = LayerManagement()
        self.show_layers()
        self.spinBox_scale.blockSignals(True)
        self.spinBox_scale.setValue(self.DEFAULT_SCALING)
        self.spinBox_scale.blockSignals(False)
        self.cur_color = (0,0,0)    # RGB
        self.set_toolBox({'spinBox_ksizeW':0, 'spinBox_ksizeH':0, 'choose_color':0, 'remove_small':0})
        self.label_x = 0
        self.label_y = 0
        self.label_image.setGeometry(QRect(self.label_x, self.label_y, self.label_w, self.label_h))
        self.label_image_scaling = 10

    def on_chooseColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            print("chosen color: ", color.name())
            self.label_color.setStyleSheet(f"background-color: {color.name()}")
            self.cur_color = color.getRgb()
            # self.cur_color = (r, g, b)
            print(f't:{type(self.cur_color)}, color:{self.cur_color}')

    def on_listWidget_layer_item_selected(self, layer_name):
        layer_type = self.layer_management.get_layer_type(layer_name)
        if layer_type == 'Image':
            self.cur_select_imgname = layer_name
            self.statusbar_label.setText(f"Current selected image: {self.cur_select_imgname}")
            print(f"Current selected image: {self.cur_select_imgname}")
            self.on_metadata_changed()
        elif layer_type == 'Mask':
            self.cur_select_maskname = layer_name
            self.statusbar_label.setText(f"Current selected mask: {self.cur_select_maskname}")
            print(f"Current selected mask: {self.cur_select_maskname}")
            self.on_metadata_changed()
        # todo: change cur_select_layername when layer deleted

    def on_layer_state_changed(self, layer_name, state):
        if self.layer_management.modify_layer_state(layer_name, state):
            txt = f"layer {layer_name} state changes to {state}"
            print(txt)
            self.statusbar.showMessage(txt)
            self.show_layers()

    def on_rescaling(self, update_layermanagement=True):
        self.img_scaling = self.spinBox_scale.value()
        self.scaled_img = cv2.resize(self.img, (int(self.img_size[0]*self.img_scaling),
                                                int(self.img_size[1]*self.img_scaling)))
        if update_layermanagement:
            self.layer_management.update_original_img(self.scaled_img)
            self.show_layers()
        txt = f"image rescale to {int(self.img_scaling*100)}%"
        self.statusbar.showMessage(txt)
        self.on_metadata_changed()

    def on_PDFReport(self):
        print('current tab index:',self.tabWidget.currentIndex())

        if self.tabWidget.currentIndex() == 0:
            return

        # load patient info
        self.report_info['patient_name'] = self.lineEdit_patientName.text()
        self.report_info['patient_age'] = self.spinBox_patientAge.value()
        self.report_info['patient_gender'] = self.comboBox_patientGender.currentText()
        self.report_info['patient_id'] = self.lineEdit_patientID.text()

        if self.cur_img is not None:
            rgb_img = cv2.cvtColor(self.cur_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(r'./report/image.png', rgb_img)
            create_report(self.report_info, r'./report/')
            self.reload_report()


    # --------------------------MouseControl------------------------------

    def mousePressEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            self.mousePressing = True
            self.label_image_offsetX = e.x()
            self.label_image_offsetY = e.y()
        print(f"mouse_press: label_img: ({self.label_image.x()},{self.label_image.y()}) {self.label_image.width()}*{self.label_image.height()}")

    def mouseReleaseEvent(self, e):
        self.mousePressing = False
        self.label_image_offsetX = ""
        self.label_image_offsetY = ""

    def mouseMoveEvent(self, e):
        if self.mousePressing:
            self.label_image_x1 = e.x()
            self.label_image_y1 = e.y()
            if self.label_image_offsetX != "" and self.label_image_offsetY != "":
                self.label_x = self.label_x + (self.label_image_x1 - self.label_image_offsetX)
                self.label_y = self.label_y + (self.label_image_y1 - self.label_image_offsetY)
        self.label_image_offsetX = self.label_image_x1
        self.label_image_offsetY = self.label_image_y1
        self.label_image_control()

    def wheelEvent(self, e):
        # print("Wheel Event")
        # pass
        self.angle = e.angleDelta() / 8
        self.angleY = self.angle.y()
        if self.angleY > 0:
            if self.LABEL_IMAGE_SCALE_MIN <= self.label_image_scaling <= self.LABEL_IMAGE_SCALE_MAX-1:
                self.label_image_scaling += 1
        elif self.angleY < 0:
            if self.LABEL_IMAGE_SCALE_MIN+1 <= self.label_image_scaling <= self.LABEL_IMAGE_SCALE_MAX:
                self.label_image_scaling -= 1
        self.label_image_control(set_pixmap=True)
        self.label_image.setPixmap(self.cur_pixmap)

    def label_image_control(self, set_pixmap=False):
        label_w_scaled = int(self.label_w * self.label_image_scaling / 10)
        label_h_scaled = int(self.label_h * self.label_image_scaling / 10)
        if set_pixmap:
            q_img = cv2_2_QImage(self.cur_img)
            self.cur_pixmap = QPixmap(q_img.scaled(label_w_scaled, label_h_scaled))
        self.label_image.setGeometry(QRect(self.label_x, self.label_y, label_w_scaled, label_h_scaled))

    def show_about_dialog(self):
        about_dialog = AboutDialog(self)
        about_dialog.exec()

    def show_tutorial_dialog(self):
        tutorial_dialog = TutorialDialog(self)
        tutorial_dialog.exec()

    # --------------------------Functionality-----------------------------

    def reload_img(self):
        self.img = cv2.imread(self.img_filepath)
        self.img_size = self.img.shape[:-1]
        self.spinBox_scale.setValue(self.DEFAULT_SCALING)
        self.on_rescaling(False)
        self.get_default_labelwh((self.scaled_img.shape[0], self.scaled_img.shape[1]))
        self.add_layer(self.filename, self.scaled_img)

        # set info
        self.report_info['image_resolution'] = self.img_size


    def reload_report(self):
        doc = fitz.open(r'./report/'+PDF_PATH)
        page = doc.load_page(0)  # 加载第 1 页
        pix = page.get_pixmap()
        img = QImage(
            pix.samples,
            pix.width,
            pix.height,
            pix.stride,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(img)
        self.label_report.setPixmap(pixmap)


    def show_layers(self):
        self.cur_img = self.layer_management.get_mixed_img()
        if self.cur_img is not None:
            self.label_image_control(set_pixmap=True)
            self.label_image.setPixmap(self.cur_pixmap)
        else:
            self.label_image.setPixmap(QPixmap())
        txt = f"Image refreshed."
        print(txt)
        self.statusbar.showMessage(txt)

    def current_selected_operation(self):
        if self.treeView_operations.selectedIndexes():
            return str(self.treeView_operations.selectedIndexes()[0].data())
        else:
            return None

    listWidget_itemSize = QSize(164, 26)
    # todo: find an appropriate size
    def add_layer(self, img_name, img, is_contour=False):
        layer_name = self.layer_management.add_layer(img_name, img, is_contour=is_contour)
        item = QListWidgetItem()
        item.setSizeHint(self.listWidget_itemSize)
        check_box = LayerCheckBox(layer_name, self.layer_management, self.listWidget_layer, self.listWidget_itemSize)
        self.listWidget_layer.addItem(item)
        self.listWidget_layer.setItemWidget(item, check_box)
        check_box.stateChanged.connect(lambda state: self.on_layer_state_changed(check_box.text(), state))
        # self.listWidget_layer.itemClicked.connect(self.on_listWidget_layer_item_clicked)
        check_box.listWidget_layer_item_selected.connect(self.on_listWidget_layer_item_selected)
        check_box.move_refresh_signal.connect(self.show_layers)
        self.show_layers()

    def get_spinBox_ksize(self):
        w = self.spinBox_ksizeW.value()
        if self.spinBox_ksizeH.isReadOnly():
            h = w
        else:
            h = self.spinBox_ksizeH.value()
        return w, h

    #---------------------OpenAI API----------------------------

    def construct_prompt(self):
        r = self.report_info
        system_prompt = """
你是一名资深肝脏病理学专家，需要根据用户提供的肝脏病理切片量化数据，结合临床知识进行分析。请按以下步骤严谨评估：

#### **1. 数据校验**
- 检查输入数据的完整性（性别、年龄、汇管区数量/面积等是否齐全）。
- 若数据异常（如汇管区总数=0或面积占比>100%），要求用户复核。

#### **2. 核心分析**
##### **（1）汇管区分布评估**
- 正常汇管区占比 = 正常个数/总个数
- 纤维化汇管区占比 = 纤维化个数/总个数

##### **（2）面积分析**
- 总汇管区面积占比（{portal_area}%）：正常值约5-10%（超过15%提示异常）
- 纤维化面积占比（{fibre_portal_area}%）：
  - <5% → "轻度纤维化（F1）"
  - 5-15% → "中度纤维化（F2-F3）"
  - >15% → "肝硬化风险（F4）"

##### **（3）年龄/性别关联**
- 男性+高龄（>50岁）+高纤维化占比 → 增加代谢性疾病（如脂肪肝）概率
- 女性+高纤维化占比 → 需排查自身免疫性肝炎（AIH）

#### **3. 输出要求**
用外行人能够看懂的简单段落回复，减少小标题数量

#### **4. 注意事项**
- 避免绝对化表述（如"确诊为肝硬化"），改用"提示可能"。
- 若数据不足（如缺乏炎症指标），明确说明局限性。
- 有可能出现数据为空的情况，这时候提示用户应当先处理出数据，再提问AI"""
        user_prompt = ("性别:{}\n年龄:{}\n汇管区个数:{}\n正常汇管区个数:{}\n纤维化汇管区个数:{}\n汇管区面积占比:{}\n正常汇管区面积占比:{}\n纤维化汇管区面积占比:{}"
                       .format(r["patient_gender"], r["patient_age"], r["portal_num"], r["normal_portal_num"],
                               r["fibre_portal_num"], r["portal_area"], r["normal_portal_area"], r["fibre_portal_area"]))
        return system_prompt, user_prompt

    def start_stream(self):
        if self.stream_thread and self.stream_thread.isRunning():
            return

        # 获取输入
        system_prompt, user_prompt = self.construct_prompt()
        api_key = self.lineEdit_apikey.text()
        base_url = self.lineEdit_baseUrl.text()
        model = self.comboBox_AIModel.currentText()

        self.stream_thread = StreamThread(
            system_prompt, user_prompt, api_key,
            base_url=base_url,
            model=model
        )
        self.stream_thread.update_signal.connect(self.update_output)
        self.stream_thread.finished.connect(self.stream_finished)
        self.stream_thread.start()

        # 更新按钮状态
        self.pushBtn_AIGenerate.setEnabled(False)
        self.pushBtn_AIStop.setEnabled(True)

    def stop_stream(self):
        """停止流式请求"""
        if self.stream_thread and self.stream_thread.isRunning():
            self.stream_thread.stop()

    def stream_finished(self):
        """流式请求完成后的清理工作"""
        self.pushBtn_AIGenerate.setEnabled(True)
        self.pushBtn_AIStop.setEnabled(False)
        self.stream_thread = None

    def update_output(self, text):
        """更新输出框内容"""
        self.textBrowser_AIOutput.setText(text)

    def closeEvent(self, event):
        if self.stream_thread and self.stream_thread.isRunning():
            self.stream_thread.stop()
            self.stream_thread.wait()
        event.accept()


def get_save_path(folder, img_name, ext='.png'):
    '''
    get the actual save directory

    :param folder: dir of the folder to save
    :param img_name: img_name(with roi id) without extension name
    :return:
    '''
    os.makedirs(folder, exist_ok=True)
    new_name = f"{img_name}{ext}"
    save_path = os.path.join(folder, new_name)

    counter = 1
    while os.path.exists(save_path):
        new_name = f"{img_name}_{counter}{ext}"
        save_path = os.path.join(folder, new_name)
        counter += 1
    return save_path


def main():
    app = QApplication(sys.argv)
    window = DemoWindow()

    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    # print(app.styleSheet())

    window.ui.show()
    sys.exit(app.exec())


if __name__=="__main__":
    main()
