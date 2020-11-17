class InputListener:
    def __init__(self, id):
        self.id = id

    def on_input_received(self, data):
        pass

    def getId(self):
        return self.id