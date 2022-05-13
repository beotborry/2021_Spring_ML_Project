import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, input_dim, hidden_size):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # define cnn model
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(1, 16, 3, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.res1_conv1 = nn.Conv2d(16, 16, 3, padding = 1, bias=False)
        self.res1_bn1 = nn.BatchNorm2d(16)
        self.res1_conv2 = nn.Conv2d(16, 16, 3, padding = 1, bias=False)
        self.res1_bn2 = nn.BatchNorm2d(16)

        self.res2_conv1 = nn.Conv2d(16, 32, 3, stride = 2, padding = 1, bias=False)
        self.res2_bn1 = nn.BatchNorm2d(32)
        self.res2_conv2 = nn.Conv2d(32, 32, 3, padding = 1, bias=False)
        self.res2_bn2 = nn.BatchNorm2d(32)
        self.res2_downsample1 = nn.Sequential(
          nn.Conv2d(16, 32, 3, stride=2, padding = 1, bias=False),
          nn.BatchNorm2d(32)
        )

        self.res3_conv1 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1, bias=False)
        self.res3_bn1 = nn.BatchNorm2d(64)
        self.res3_conv2 = nn.Conv2d(64, 64, 3, padding = 1, bias=False)
        self.res3_bn2 = nn.BatchNorm2d(64)
        self.res3_downsample1 = nn.Sequential(
          nn.Conv2d(32, 64, 3, stride = 2, padding = 1, bias=False),
          nn.BatchNorm2d(64)
        )        
        self.fc = nn.Linear(3136, self.hidden_size)
        self.dropout = nn.Dropout(0.4)
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        inputs: (Batch x Sequence_length, Channel=1, Height, Width)
        outputs: (Sequence_length, Batch_size, Hidden_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1: code CNN forward path
        batch_size = int(inputs.shape[0] / 10)

        x = self.conv1(inputs)
        x = self.bn1(x)


# 16 -> 16
        res1 = x
        x = self.res1_conv1(x)
        x = self.res1_bn1(x)
        x = self.relu(x)
        x = self.res1_conv2(x)
        x = self.res1_bn2(x)
        x = x + res1
        x = self.relu(x)

# 16 -> 32
        res2 = x
        x = self.res2_conv1(x)
        x = self.res2_bn1(x)
        x = self.relu(x)
        x = self.res2_conv2(x)
        x = self.res2_bn2(x)
        res2 = self.res2_downsample1(res2)
        x = x + res2
        x = self.relu(x)

# 32 -> 64
        res3 = x
        x = self.res3_conv1(x)
        x = self.res3_bn1(x)
        x = self.relu(x)
        x = self.res3_conv2(x)
        x = self.res3_bn2(x)
        res3 = self.res3_downsample1(res3)
        x = x + res3
        x = self.relu(x)


        x = x.reshape(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)

        outputs = x.reshape(batch_size, 10, -1).permute(1, 0, 2)
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()

        # define the properties
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # lstm cell
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2: Define input, output projection layer to fit dimension
        # output fully connected layer to project to the size of the class
        self.fc_in_cnn = nn.Linear(self.input_dim, self.hidden_size)
        self.fc_in_other = nn.Linear(26, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, feature, h, c):
        """
        feature: (Sequence_length, Batch_size, Input_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem3: Design LSTM model for letter sorting
        # NOTE: sequence length of feature can be various
        x = self.fc_in_cnn(feature) if feature.shape[2] != 26 else self.fc_in_other(feature) 
        
        output, (h_next, c_next) = self.lstm(x, (h, c))
        output = self.fc_out(output)
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return output, h_next, c_next  # (sequence_lenth, batch, num_classes), (num_rnn_layers, batch, hidden_dim), (num_rnn_layers, batch, hidden_dim)
