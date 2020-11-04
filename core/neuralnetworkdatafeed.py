from input_module.input_listener import InputListener


class NeuralNetworkDataFeed(InputListener):
    def __init__(self):
        super(NeuralNetworkDataFeed, self).__init__("NeuralNetworkDataFeed")
    
    def on_input_received(self, data):
        # feed neural network with data parsed by input processor
        file = open("data_output/output2.csv", "w")
        for row in data:
            file.write(row)