

class Disease:
    def __init__(self, tamagotchi_app, initial_level=0):
        self.tamagotchi_app = tamagotchi_app
        self.d_level = initial_level



    def increase_level(self):
        if self.tamagotchi_app.tamagotchi_state['illness'] <= 10:
            self.d_level += 0.1


    def decrease_level(self):
        if self.tamagotchi_app.tamagotchi_state['illness'] <= 20 or "under the meds effect":
            self.d_level -= 0.1
            if self.d_level < 0:
                self.d_level = 0


    def get_level(self):
        return self.d_level

    def apply_effects(self, tamagotchi_app):
        if self.d_level > 70:
            ("Мёртв")
            tamagotchi_app.tamagotchi_state['health'] = 0
        if 50 < self.d_level <= 70:
            "Средняя форма"
            tamagotchi_app.tamagotchi_state['happiness'] -= tamagotchi_app.happiness * self.d_level
            tamagotchi_app.tamagotchi_state['mental'] -= tamagotchi_app.mental * self.d_level
        if 0 < self.d_level <= 50:
            "Лёгкая форма"
            tamagotchi_app.tamagotchi_state['health'] -= tamagotchi_app.health * self.d_level
            tamagotchi_app.tamagotchi_state['phys'] -= tamagotchi_app.phys * self.d_level


class Cold:
    def __init__(self):

