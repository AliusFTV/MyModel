import json as js

class Item:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {'name': self.name}

class ButtonState:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_dict(self):
        return {'x': self.x, 'y': self.y}

class SaveManager:
    def __init__(self, tamagotchi_app):
        self.tamagotchi_app = tamagotchi_app
        self.buttons = self.tamagotchi_app.buttons
        global save_path
        save_path = "C:/Users/alius/PycharmProjects/pythonProject/tg_save.json"

    def save_state(self, tamagotchi_state, inventory, buttons):
        state_to_save = dict(
            tg_state=tamagotchi_state,
            inventory=[item.to_dict() for item in inventory],
            button_positions=self.save_button_positions(buttons)
        )
        with open(save_path, "w") as file:
            js.dump(state_to_save, file)

    def load_save(self, default_state):
        try:
            with open(save_path, "r") as file:
                saved_state = js.load(file)
                tamagotchi_state = saved_state.get('tg_state', default_state)
                inventory_data = saved_state.get('inventory', [])
                inventory = [Item(**item_data) for item_data in inventory_data]
                button_positions = saved_state.get('button_positions', [])
                return tamagotchi_state, inventory, button_positions
        except FileNotFoundError:
            tamagotchi_state = default_state
            inventory = []
            return tamagotchi_state, inventory
    def save_button_positions(self, buttons):
        return [ButtonState(button.x(), button.y()).to_dict() for button in buttons]

    def load_button_positions(self, positions):
        for DraggableButton, position in zip(self.buttons, positions):
            DraggableButton.move(position['x'], position['y'])