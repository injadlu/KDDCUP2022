import torch
import torch.nn as nn
import time

class Encoder(nn.Module):
    """
    Desc:
        A simple GRU model to encode
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(Encoder, self).__init__()
        self.hidC = settings["in_var"]
        self.hidR = settings["hidR"]
        self.out = settings["out_var"]
        # self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(self.hidC, self.hidR, settings["lstm_layer"])
        # (input, hidden, layer)

        self.init_weight()
    # all-0

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)


    def forward(self, x_enc):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        # time major : [L, B, D]
        input = x_enc
        input = input.permute(1, 0, 2)
        dec, hidden = self.lstm(input)
        return hidden

class Decoder(nn.Module):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(Decoder, self).__init__()
        self.output_len = 144
        self.hidC = settings["in_var"]
        self.hidR = settings["hidR"]
        self.out = settings["out_var"]
        # self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(self.hidC, self.hidR, settings["lstm_layer"])
        self.projection = nn.Linear(self.hidR, self.out)
        self.init_weight()
    # all-0

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
        for name, param in self.projection.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x_dec, hidden):
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        # time major : [L, B, D]
        input = x_dec
        input = input.permute(1, 0, 2)
        dec, _ = self.lstm(input, hidden)
        dec = dec.permute(1, 0, 2)
        dec = self.projection(dec)
        dec = dec[:, -self.output_len:, -self.out:]
        return dec  # [B, L, D]

class HalfGruModel(nn.Module):
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(HalfGruModel, self).__init__()
        self.encoder = Encoder(settings)
        self.decoder = Decoder(settings)
    def forward(self, x_enc):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        # time major : [L, B, D]
        out_len = 144
        hidden = self.encoder(x_enc)
        x_dec = torch.zeros([x_enc.shape[0], out_len, x_enc.shape[2]], dtype=torch.float).to(x_enc.device)
        dec = self.decoder(x_dec, hidden)
        return dec