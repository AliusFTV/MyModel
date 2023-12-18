from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtGui import QIcon

class SystemTrayApp(QApplication):
    def __init__(self, sys_argv, tamagotchi_app):
        super(SystemTrayApp, self).__init__(sys_argv)
        self.tamagotchi_app = tamagotchi_app

        self.tray_icon = QSystemTrayIcon(QIcon("sys.png"), self)

        self.tray_menu = QMenu()
        self.show_action = QAction("Показать", self)
        self.exit_action = QAction("Закрыть", self)
        self.show_action.triggered.connect(self.show_main_window)
        self.exit_action.triggered.connect(self.quit)
        self.tray_menu.addAction(self.show_action)
        self.tray_menu.addAction(self.exit_action)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self.toggle_window)

        self.tray_icon.show()

    def show_main_window(self):
        self.tamagotchi_app.show_main_window()

    def toggle_window(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            if self.tamagotchi_app.isHidden():
                self.tamagotchi_app.show_main_window()
            else:
                self.tamagotchi_app.hide_main_window()



