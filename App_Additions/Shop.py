from PyQt5.QtCore import Qt
from SAL import Item
from App_Additions.custom_menu import Labels_Buttons_BG_FG, DraggableButton, DraggableLabel


class Shop:
    def __init__(self, tamagotchi_app):
        label_and_button_style = Labels_Buttons_BG_FG.get_label_and_button_style()
        self.tamagotchi_app = tamagotchi_app
        self.save_manager = tamagotchi_app.save_manager
        self.inventory = tamagotchi_app.inventory
        self.label_shop = DraggableLabel("Добро пожаловать в магазин!", tamagotchi_app)
        self.label_shop.setAlignment(Qt.AlignCenter)
        tamagotchi_app.v_layout.addWidget(self.label_shop)
        self.label_shop.setStyleSheet(label_and_button_style)
        self.label_shop.setFixedSize(300, 25)

        self.button_buy_mug = DraggableButton("Купить миску (10 монет)", tamagotchi_app)
        if "Миска" in [item.name for item in self.tamagotchi_app.inventory]:
            self.button_buy_mug.setEnabled(False)
        self.button_buy_mug.clicked.connect(self.buy_mug)
        tamagotchi_app.v_layout.addWidget(self.button_buy_mug)
        self.button_buy_mug.setStyleSheet(label_and_button_style)
        self.button_buy_mug.setFixedSize(400, 30)


        self.button_back = DraggableButton("Назад", tamagotchi_app)
        self.button_back.clicked.connect(self.return_to_main)
        tamagotchi_app.v_layout.addWidget(self.button_back)
        self.button_back.setStyleSheet(label_and_button_style)
        self.button_back.setFixedSize(400, 30)

    def buy_mug(self):
        if self.tamagotchi_app.tamagotchi_state['coins'] >= 10:
            if "Миска" not in [item.name for item in self.tamagotchi_app.inventory]:
                self.tamagotchi_app.tamagotchi_state['coins'] -= 10
                self.inventory.append(Item("Миска"))
                self.tamagotchi_app.update_labels()
                self.tamagotchi_app.show_info_message("Миска успешно приобретена!", self.button_buy_mug)
                self.update_buy_mug_button()
        else:
            self.tamagotchi_app.show_info_message("Недостаточно монет для покупки миски!", self.button_buy_mug)

    def return_to_main(self):
        self.label_shop.hide()
        self.button_buy_mug.hide()
        self.button_back.hide()
        for widget in self.tamagotchi_app.labels + self.tamagotchi_app.buttons:
            widget.show()

        self.tamagotchi_app.setup_main_window()

    def update_buy_mug_button(self):
        if any(item.name == "Миска" for item in self.tamagotchi_app.inventory):
            self.button_buy_mug.setEnabled(False)
        else:
            self.button_buy_mug.setEnabled(True)

