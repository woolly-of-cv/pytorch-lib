from torch import nn

class View(nn.Module):
    def __init__(self, out_dim):
        super(View, self).__init__()
        self.out_dim = out_dim

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = input.view(-1, self.out_dim)

        return out