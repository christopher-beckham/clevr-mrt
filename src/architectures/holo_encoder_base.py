import torch
from torch import nn

class RaiseException(nn.Module):
    def __init__(self):
        super(RaiseException, self).__init__()
    def forward(self, x):
        raise Exception("cam_encode must be replaced with a module")

class HoloEncoderBase(nn.Module):
    def __init__(self):
        super(HoloEncoderBase, self).__init__()

        # must be implemented
        self.cam_encode = RaiseException()

    def encode(self, x):
        # encode x into z
        raise NotImplementedError(
            "encode() needs to be implemented"
        )

    def enc2vol(self, x):
        # convert z into h
        raise NotImplementedError(
            "enc2vol() needs to be implemented"
        )

    @property
    def postprocessor(self):
        # This is used to return the module (or a list of
        # modules) which comprise the postprocessor so
        # that we can easily isolate it if need be.
        raise NotImplementedError(
            "This needs to return the postprocessor module"
        )

    def postprocess(self, x):
        raise NotImplementedError(
            "postprocess() needs to be implemented"
        )

    def coord_map_3d(self, shape, start=-1, end=1):
        if type(shape) == int:
            m = n = o = shape
        else:
            m, n, o = shape
        x_coord_row = torch.linspace(start, end, steps=o).\
            type(torch.cuda.FloatTensor)
        y_coord_row = torch.linspace(start, end, steps=n).\
            type(torch.cuda.FloatTensor)
        z_coord_row = torch.linspace(start, end, steps=m).\
            type(torch.cuda.FloatTensor)

        x_coords = x_coord_row.unsqueeze(0).\
            expand(torch.Size((m, n, o))).unsqueeze(0)
        y_coords = y_coord_row.unsqueeze(1).\
            expand(torch.Size((m, n, o))).unsqueeze(0)
        #z_coords = z_coord_row.unsqueeze(2).expand(torch.Size((m, n, o))).unsqueeze(0)
        z_coords = z_coord_row.unsqueeze(0).\
            view(-1, m, 1, 1).repeat(1, 1, n, o)
        return torch.cat([x_coords, y_coords, z_coords], 0)

    def coord_map(self, shape, start=-1, end=1):
        """
        Gives, a 2d shape tuple, returns two mxn coordinate maps,
        Ranging min-max in the x and y directions, respectively.
        """
        m = shape
        n = shape
        x_coord_row = torch.linspace(start, end, steps=n).\
            type(torch.cuda.FloatTensor)
        y_coord_row = torch.linspace(start, end, steps=m).\
            type(torch.cuda.FloatTensor)
        x_coords = x_coord_row.unsqueeze(0).\
            expand(torch.Size((m, n))).unsqueeze(0)
        y_coords = y_coord_row.unsqueeze(1).\
            expand(torch.Size((m, n))).unsqueeze(0)
        return torch.cat([x_coords, y_coords], 0)
