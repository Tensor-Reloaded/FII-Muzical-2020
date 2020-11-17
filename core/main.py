from transformer import Transformer
from encoder import Encoder
from decoder import Decoder
from transformerbuilder import TransformerBuilder
from input_module.input_processor import InputProcessor
from neuralnetworkdatafeed import NeuralNetworkDataFeed

def main():
    neural_network_data_feeder = NeuralNetworkDataFeed()

    input_processor = InputProcessor()
    input_processor.register_input_listener(neural_network_data_feeder)

    input_processor.process_one("samples/input_1.mid")

    input_processor.unregister_input_listener(neural_network_data_feeder)

    # transformer = TransformerBuilder()\
    #     .with_encoder(Encoder())\
    #     .with_decoder(Decoder())\
    #     .build()


if __name__ == "__main__":
    main()