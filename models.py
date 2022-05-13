import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import LSTM, CustomCNN


class ConvLSTM(nn.Module):
    def __init__(self, sequence_length, num_classes, batch_size, cnn_layers=None,
                 cnn_input_dim=1, rnn_input_dim=256,
                 cnn_hidden_size=256, rnn_hidden_size=512, rnn_num_layers=1, rnn_dropout=0.0):
        # NOTE: you can freely add hyperparameters argument
        super(ConvLSTM, self).__init__()

        # define the properties, you can freely modify or add hyperparameters
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_input_dim = cnn_input_dim
        self.rnn_input_dim = rnn_input_dim
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.batch_size = batch_size
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.conv = CustomCNN(input_dim=self.cnn_input_dim, hidden_size=self.cnn_hidden_size)
        self.lstm = LSTM(input_dim=self.rnn_input_dim, hidden_size=self.rnn_hidden_size, vocab_size=26, num_layers=self.rnn_num_layers)
        # NOTE: you can define additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        input is (imgaes, labels) (training phase) or images (test phase)
        images: sequential features of (Batch x Sequence_length, Channel=1, Height, Width)
        labels: (Batch x Sequence_length,)
        outputs should be a size of (Batch x Sequence_length, Num_classes)
        """

        # for teacher-forcing
        have_labels = False
        if len(inputs) == 2:
            have_labels = True
            images, labels = inputs
        else:
            images = inputs

        batch_size = images.size(0) // self.sequence_length
        hidden_state = torch.zeros((self.rnn_num_layers, batch_size, self.rnn_hidden_size)).to(images.device)
        cell_state = torch.zeros((self.rnn_num_layers, batch_size, self.rnn_hidden_size)).to(images.device)

        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem4: input image into CNN and RNN sequentially.
        # NOTE: you can use teacher-forcing using labels or not
        lstm_input = self.conv(images)
        outputs, hidden_state, cell_state = self.lstm(lstm_input, hidden_state, cell_state)

        if have_labels:
          labels = labels.reshape(self.batch_size, self.sequence_length).permute(1,0)

          lstm_input = torch.ones((1, self.batch_size, self.num_classes)).to(images.device)
          output, hidden_state, cell_state = self.lstm(lstm_input, hidden_state, cell_state)

          outputs[0] = output

          for i in range(self.sequence_length - 1):

            is_tf = True if torch.randint(high = 10, size = (1,)).item() <= 4 else False
            if is_tf:
              lstm_input = torch.eye(self.num_classes)[labels[i]].reshape(1, -1, self.num_classes).float().to(images.device)
            else: lstm_input = torch.eye(self.num_classes)[output.argmax(dim=2)].float().to(images.device)

            output, hidden_state, cell_state = self.lstm(lstm_input, hidden_state, cell_state)
            outputs[i+1] = output
      
        else:
          lstm_input = torch.ones((1, self.batch_size, self.num_classes)).to(images.device)
          output, hidden_state, cell_state = self.lstm(lstm_input, hidden_state, cell_state)

          outputs[0] = output
          
          for i in range(self.sequence_length - 1):
            lstm_input = torch.eye(self.num_classes)[output.argmax(dim=2)].float().to(images.device)
            output, hidden_state, cell_state = self.lstm(lstm_input, hidden_state, cell_state)
            outputs[i + 1] = output

           
        outputs = outputs.permute(1,0,2).reshape(-1, self.num_classes)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs  # (BxT, N_class)
