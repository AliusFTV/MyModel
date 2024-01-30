from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QLabel


class DraggableButton(QPushButton):
    def __init__(self, text, parent=None):
        super(DraggableButton, self).__init__(text, parent)
        self.setMouseTracking(True)
        self.mousePressPosition = None
        self.mouseMovePosition = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mousePressPosition = event.globalPos()
            self.mouseMovePosition = event.globalPos()
            super(DraggableButton, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            if self.mousePressPosition:
                delta = event.globalPos() - self.mouseMovePosition
                self.move(self.pos() + delta)
                self.mouseMovePosition = event.globalPos()

    def mouseReleaseEvent(self, event):
        if self.mousePressPosition is not None:
            moved = event.globalPos() - self.mousePressPosition
            if moved.manhattanLength() > 3:
                event.ignore()
                return

        super(DraggableButton, self).mouseReleaseEvent(event)

    def get_position(self):
        return {'x': self.x, 'y': self.y}
class DraggableLabel(QLabel):
    def __init__(self, text, parent=None):
        super(DraggableLabel, self).__init__(text, parent)
        self.setMouseTracking(True)
        self.mousePressPosition = None
        self.mouseMovePosition = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mousePressPosition = event.globalPos()
            self.mouseMovePosition = event.globalPos()
            super(DraggableLabel, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            if self.mousePressPosition:
                delta = event.globalPos() - self.mouseMovePosition
                self.move(self.pos() + delta)
                self.mouseMovePosition = event.globalPos()

    def mouseReleaseEvent(self, event):
        if self.mousePressPosition is not None:
            moved = event.globalPos() - self.mousePressPosition
            if moved.manhattanLength() > 3:
                event.ignore()
                return

        super(DraggableLabel, self).mouseReleaseEvent(event)

class Labels_Buttons_BG_FG:
    @staticmethod
    def apply_style_to_labels_and_buttons(labels, buttons):
        style = """
            QLabel {
            background-color: #822;
                color: white;
                border: 1.5px solid white;
                padding: 1px;
            }
            QPushButton {
                background-color: #555;
                color: white;
                border: 2px solid white;
                padding: 0;
                margin:0;
                position: absolute;
                left: 0;
                right: 100px;
                bottom: 0;
                top: 0;
                
            }
            QPushButton:hover {
                background-color: #477;
            }
        """
        for label in labels:
            label.setStyleSheet(style)

        for button in buttons:
            button.setStyleSheet(style)
    @staticmethod
    def get_label_and_button_style():
        return """
            QLabel {
            background-color: #822;
                color: white;
                padding: 1px;
            }
            QPushButton {
                background-color: #555;
                color: white;
                border: 1px solid white;
                padding: 1px;
                margin: 1px;
            }
            QPushButton:hover {
                background-color: #477;
            }
        """