import torch


class PointWiseFeedForwardNetwork:
    def __init__(self, dimension_model, dimension_feed_forward):
        self.sequential_model = torch.nn.Sequential(
            torch.nn.Linear(dimension_model, dimension_feed_forward),
            torch.nn.ReLU(),
            torch.nn.Linear(dimension_feed_forward, dimension_model)
        )

    def get_model(self, data):
        return self.sequential_model(data)