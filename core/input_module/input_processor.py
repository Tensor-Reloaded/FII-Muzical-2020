import py_midicsv as midi_reader

class InputProcessor:
    def __init__(self):
        self.listeners = []

    def register_input_listener(self, input_listener):
        print("Registering input listener with id {}".format(input_listener.getId()))
        self.listeners.append(input_listener)

    def unregister_input_listener(self, input_listener):
        for listener in self.listeners:
            if listener.getId() == input_listener.getId():
                print("Removing listener with id {}".format(input_listener.getId()))
                self.listeners.remove(listener)
                return

        print("No listener found for id {}".format(input_listener.getId()))
        return

    def process_next(self):
        # todo use library to process input
        data = "test"
        self.notify_listeners(data)

    def process_one(self, file_path):
        data = midi_reader.midi_to_csv(file_path)
        self.notify_listeners(data)

    def notify_listeners(self, data):
        for listener in self.listeners:
            listener.on_input_received(data)