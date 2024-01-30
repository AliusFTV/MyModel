from PyQt5.QtCore import QThread, pyqtSignal

class TamagotchiWorker(QThread):
    update_signal = pyqtSignal()

    def __init__(self, tamagotchi_app):
        super().__init__()
        self.tamagotchi_app = tamagotchi_app

    def run(self):
        while True:
            self.tamagotchi_app.update_tamagotchi_state_thread()
            self.update_signal.emit()
            self.msleep(10000)