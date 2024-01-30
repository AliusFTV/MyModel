import threading
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from tamagotchi_worker import TamagotchiWorker
from time_sync import TimeSync
import time
from SAL import SaveManager
from App_Additions.Shop import Shop
from system_tray_app import SystemTrayApp
from App_Additions.custom_menu import Labels_Buttons_BG_FG, DraggableButton, DraggableLabel, QPushButton


class TamagotchiApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.showFullScreen()
        self.start_time = 0
        self.inventory = []
        self.labels = []
        self.buttons = []
        self.original_widgets = []
        self.shop = None
        self.button_positions = None
        self.default_state = dict(hunger=100, happiness=100, health=100, mental=50, phys=50, illness=10, coins=1000)
        self.tamagotchi_state = self.default_state
        self.tamagotchi_worker = TamagotchiWorker(self)
        self.save_manager = SaveManager(self)

        self.time_sync = TimeSync()
        self.state_lock = threading.Lock()
        self.setup_main_window()
        self.create_labels()
        self.update_tamagochi_state_thread()
        self.button()
        self.v_layout = QVBoxLayout(self)
        self.v_layout.setSpacing(0)
        Labels_Buttons_BG_FG.apply_style_to_labels_and_buttons(self.labels, self.buttons)
        for label in self.labels:
            self.v_layout.addWidget(label)
            label.setFixedSize(300, 25)
        for button in self.buttons:
            self.v_layout.addWidget(button)
            button.setFixedSize(400, 30)
        self.load_save()

    def sync_time(self):
        formatted_time = self.time_sync.sync_time()
        self.label_sync_time.setText(formatted_time)

    def get_current_time(self):
        self.sync_time()
        return time.time() - self.time_sync.start_time

    def create_labels(self):
        self.label_sync_time = DraggableLabel("")
        self.labels.append(self.label_sync_time)

        self.label_coins = DraggableLabel(f"Монеты: {self.tamagotchi_state['coins']}")
        self.labels.append(self.label_coins)

        self.label_hunger = DraggableLabel(f"Голод: {self.tamagotchi_state['hunger']}")
        self.labels.append(self.label_hunger)

        self.label_health = DraggableLabel(f"Здоровье: {self.tamagotchi_state['health']}")
        self.labels.append(self.label_health)

        self.label_happiness = DraggableLabel(f"Счастье: {self.tamagotchi_state['happiness']}")
        self.labels.append(self.label_happiness)

        self.label_mental = DraggableLabel(f"Настроение: {self.tamagotchi_state['mental']}")
        self.labels.append(self.label_mental)

        self.label_phys = DraggableLabel(f"Физуха: {self.tamagotchi_state['phys']}")
        self.labels.append(self.label_phys)

        self.label_illness = DraggableLabel(f"Иммунитет: {self.tamagotchi_state['illness']}")
        self.labels.append(self.label_illness)

    def start_update_thread(self):
        self.update_thread = threading.Thread(target=self.update_tamagochi_state_thread)
        self.update_thread.daemon = True
        self.update_thread.start()

    def update_tamagochi_state_thread(self):
        with self.state_lock:
            self.tamagotchi_state['hunger'] -= 0.01
            self.tamagotchi_state['happiness'] -= 0.03
            self.tamagotchi_state['mental'] -= 0.05
            self.tamagotchi_state['phys'] -= 0.01
            self.tamagotchi_state['illness'] -= 0.03
        self.update_labels()
        self.get_current_time()
        QTimer.singleShot(10000, self.update_tamagochi_state_thread)

    def update_labels(self):
        self.label_hunger.setText(f"Голод: {round(self.tamagotchi_state['hunger'], 2)}")
        self.label_health.setText(f"Здоровье: {round(self.tamagotchi_state['health'], 2)}")
        self.label_happiness.setText(f"Счастье: {round(self.tamagotchi_state['happiness'], 2)}")
        self.label_mental.setText(f"Настроение: {round(self.tamagotchi_state['mental'], 2)}")
        self.label_phys.setText(f"Физуха: {round(self.tamagotchi_state['phys'], 2)}")
        self.label_illness.setText(f"Иммунитет: {round(self.tamagotchi_state['illness'], 2)}")
        self.label_coins.setText(f"Монеты: {round(self.tamagotchi_state['coins'], 2)}")

    def feed(self, action, hunger, health, happiness, mental):
        if action == "put in bowl":
            has_bowl = any(item.name == "Миска" for item in self.inventory)
            if not has_bowl:
                self.show_info_message("У тебя нет миски!", self.button_feed3)
        self.tamagotchi_state['hunger'] += hunger
        self.tamagotchi_state['health'] += health
        self.tamagotchi_state['happiness'] += happiness
        self.tamagotchi_state['mental'] += mental
        self.update_labels()

    def button(self):
        self.button_feed1 = QPushButton("Бросить еду")
        self.button_feed1.setObjectName("feed_rough")
        self.button_feed1.clicked.connect(lambda: self.feed("throw", 5, -3, -6, -5))
        self.buttons.append(self.button_feed1)

        self.button_feed2 = DraggableButton("Покормить с руки")
        self.button_feed2.setObjectName("feed_hand")
        self.button_feed2.clicked.connect(lambda: self.feed("feed from hand", 5, -3, 5, 4))
        self.buttons.append(self.button_feed2)

        self.button_feed0 = DraggableButton("Ударить едой?")
        self.button_feed0.setObjectName("hit")
        self.button_feed0.clicked.connect(lambda: self.feed("hit", 0, -6, -5, -10))
        self.buttons.append(self.button_feed0)

        self.button_feed3 = DraggableButton("Положить еду в миску")
        self.button_feed3.setObjectName("feed_bowl")
        self.button_feed3.clicked.connect(lambda: self.feed("put in bowl", 10, -3, 0, 0))
        self.buttons.append(self.button_feed3)

        self.button_shop = DraggableButton("Магазин")
        self.button_shop.setObjectName("shop")
        self.button_shop.clicked.connect(self.show_shop_widgets)
        self.buttons.append(self.button_shop)

        self.button_break_bowl = DraggableButton("Разбить миску")
        self.button_break_bowl.setObjectName("break")
        self.button_break_bowl.clicked.connect(self.break_bowl)
        self.buttons.append(self.button_break_bowl)

        self.button_tray = DraggableButton("Свернуть в Трэй")
        self.button_tray.setObjectName("tray")
        self.button_tray.clicked.connect(self.hide_main_window)
        self.buttons.append(self.button_tray)

        self.button_exit = DraggableButton("Выход")
        self.button_exit.setObjectName("exit")
        self.button_exit.clicked.connect(self.on_exit)
        self.buttons.append(self.button_exit)


    def show_shop_widgets(self):
        global shop
        shop = Shop(self)
        for widget in self.labels + self.buttons:
            widget.hide()

    def break_bowl(self):
        has_bowl = any(item.name == "Миска" for item in self.inventory)
        if has_bowl:
            self.inventory = [item for item in self.inventory if item.name != "Миска"]
            self.show_info_message("Миска успешно разбита!", self.button_break_bowl)
        else:
            self.show_info_message("У вас нет миски в инвентаре!", self.button_break_bowl)
        self.update_labels()

    def create_system_tray(self):
        self.system_tray = SystemTrayApp([], self)
    def show_main_window(self):
        self.show()

    def hide_main_window(self):
        self.hide()

    def setup_main_window(self):
        for label in self.labels:
            label.show()
        for button in self.buttons:
            button.show()
    def show_info_message(self, message, button):
        original_text = button.text()
        button.setText("")
        button.setStyleSheet("background-color: lightgreen")
        button.setText(message)
        QTimer.singleShot(1000, lambda: self.restore_button_text(button, original_text))
    def restore_button_text(self, button, original_text):
        button.setText(original_text)
        button.setStyleSheet("")

    def save_state(self):
        self.button_positions = [{'x': button.x(), 'y': button.y(), 'objectName': button.objectName()} for button in
                                self.buttons]
        self.save_manager.save_state(self.tamagotchi_state, self.inventory, self.button_positions)

    def on_exit(self):
        self.save_state()
        self.close()

    def load_save(self):
        self.tamagotchi_state, self.inventory, self.button_positions = self.save_manager.load_save(self.default_state)
        self.save_manager.load_button_positions(self.button_positions, self.buttons)

if __name__ == "__main__":
    tamagotchi_app = TamagotchiApp()
    tamagotchi_app.show()



