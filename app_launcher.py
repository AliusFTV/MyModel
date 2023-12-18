from PyQt5.QtWidgets import QApplication
from main import TamagotchiApp

if __name__ == "__main__":
    app = QApplication([])
    tamagotchi_app = TamagotchiApp()
    tamagotchi_app.load_save()
    tamagotchi_app.show()
    tamagotchi_app.create_system_tray()
    app.exec ()