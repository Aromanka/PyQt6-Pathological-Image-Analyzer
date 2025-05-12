from PyQt6.QtWidgets import (QCheckBox, QMenu, QListWidget, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QVBoxLayout, QDialog)
from PyQt6.QtGui import QAction, QImage, QPaintEvent, QPainter, QPixmap, QDesktopServices
from PyQt6.QtCore import pyqtSignal, QSize, QPoint, Qt, QEvent, QUrl, QThread
from LayerManagement import LayerManagement
from openai_analyze import ask_AI
import cv2


class LayerCheckBox(QCheckBox):
    move_refresh_signal = pyqtSignal()
    listWidget_layer_item_selected = pyqtSignal(str)

    def __init__(self, name : str, lm : LayerManagement, list_widget : QListWidget, size : QSize):
        super().__init__(name)
        self.setChecked(True)
        self.menu = QMenu(self)
        self.lm = lm
        self.list_widget = list_widget

        action_select = QAction("选择", self)
        action_moveup = QAction("上移", self)
        action_movedown = QAction("下移", self)
        action_delete = QAction("删除", self)

        self.menu.addAction(action_select)
        self.menu.addSeparator()
        self.menu.addAction(action_moveup)
        self.menu.addAction(action_movedown)
        self.menu.addAction(action_delete)

        action_select.triggered.connect(self.on_select)
        action_moveup.triggered.connect(self.on_moveup)
        action_movedown.triggered.connect(self.on_movedown)
        action_delete.triggered.connect(self.on_delete)

        self.setFixedSize(size)

    def contextMenuEvent(self, a0):
        self.menu.exec(a0.globalPos())

    def find_item_index(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if self.list_widget.itemWidget(item) == self:
                return i
        return -1

    def on_moveup(self):
        index = self.find_item_index()
        if index > 0 and self.lm.move_up(self.text()):
            self.swap_items(index, index - 1)
            self.move_refresh_signal.emit()

    def on_movedown(self):
        index = self.find_item_index()
        if index < self.list_widget.count() - 1 and self.lm.move_down(self.text()):
            self.swap_items(index, index + 1)
            self.move_refresh_signal.emit()

    def swap_items(self, index1, index2):
        item1 = self.list_widget.item(index1)
        item2 = self.list_widget.item(index2)

        widget1 = self.list_widget.itemWidget(item1)
        widget2 = self.list_widget.itemWidget(item2)

        if widget1 and widget2:
            tmp_text = widget1.text()
            widget1.setText(widget2.text())
            widget2.setText(tmp_text)

            tmp_state = widget1.checkState()
            widget1.setCheckState(widget2.checkState())
            widget2.setCheckState(tmp_state)

    # mouse event overload
    def mousePressEvent(self, event):
        if event.pos().x() < 20:
            super().mousePressEvent(event)
        else:
            self.on_select()
            event.ignore()

    def on_select(self):
        self.listWidget_layer_item_selected.emit(self.text())

    def on_delete(self):
        if self.lm.remove_layer(self.text()):         # remove from layer management, and check whether it's movable
            i = self.find_item_index()
            self.list_widget.removeItemWidget(self.list_widget.takeItem(i))     # remove from list widget
            self.move_refresh_signal.emit()           # refresh the layers



def cv2_2_QImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    return QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()


def cv2_2_pixmap(img):
    return QPixmap(cv2_2_QImage(img))


class ImageLabel(QLabel):
    def __init__(self):
        super(ImageLabel, self).__init__()
        self.flag = False
        self.setMouseTracking(True)

    def mousePressEvent(self, ev):
        print("mouse press label")
        if ev.buttons() == Qt.MouseButton.LeftButton:
            self.flag = True

    def mouseMoveEvent(self, e):
        print("mouz move label")


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于本软件")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()

        self.text = ""
        with open('software_info.txt', 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                self.text += line + '\n'
        text_label = QLabel(self.text)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)

        button_layout = QHBoxLayout()

        # GitHub link
        github_button = QPushButton("GitHub")
        github_button.clicked.connect(self.open_github)
        button_layout.addWidget(github_button)

        # Close dialog
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def open_github(self):
        QDesktopServices.openUrl(QUrl("https://github.com/Aromanka/PyQt6-Pathological-Image-Analyzer"))


class TutorialDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("教程")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()

        self.text = ""
        with open('tutorial.txt', 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                self.text += line + '\n'
        text_label = QLabel(self.text)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)

        # Close dialog
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)


class StreamThread(QThread):
    """用于在后台线程中处理API流式响应的类"""
    update_signal = pyqtSignal(str)  # 用于发送更新信号的信号

    def __init__(self, system_prompt, user_prompt, api_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat"):
        super().__init__()
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._is_running = True

    def run(self):
        stream = ask_AI(self.system_prompt, self.user_prompt, self.api_key, self.base_url, self.model)

        print(f'ask: system:{self.system_prompt}, user:{self.user_prompt}, model:{self.model}, api:{self.api_key}')

        full_response = ""
        for chunk in stream:
            if not self._is_running:
                break
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                self.update_signal.emit(full_response)  # 发送更新信号

    def stop(self):
        self._is_running = False
        self.quit()

