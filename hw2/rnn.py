import numpy as np

from layers import *
from rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.
    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.
    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.
        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V
        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an fc transformation to compute the initial hidden state         #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (T, N, H).    #
        # (4) Use a (temporal) fc transformation to compute scores over the        #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################

        # Forward Pass
        # (1)

        h0, h0_cache = fc_forward(features, W_proj, b_proj) # (N, H)

        # (2)
        
        xlist, xlist_cache = word_embedding_forward(captions_in, W_embed) # (N, T-1, W)
        xlist_swap = np.swapaxes(xlist, 0, 1)  # (T-1, N, W)

        # (3)

        if self.cell_type == 'rnn':
            hlist, hlist_cache = rnn_forward(xlist_swap, h0, Wx, Wh, b) # (T-1, N, H)
        elif self.cell_type == 'lstm':
            hlist, hlist_cache = lstm_forward(xlist_swap, h0, Wx, Wh, b)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)
        hlist_swap = np.swapaxes(hlist, 0, 1) # (N, T-1, H)
        
        # (4)

        scores, socres_cache = temporal_fc_forward(hlist_swap, W_vocab, b_vocab) # (N, T-1, V)
        
        # (5)
        
        loss, dscores = temporal_softmax_loss(scores, captions_out, mask)

        # Gradients
        # (4)

        dhlist_swap, dW_vocab, db_vocab = temporal_fc_backward(dscores, socres_cache)
        
        # (3)
        
        dhlist = np.swapaxes(dhlist_swap, 0, 1)
        if self.cell_type == 'rnn':
            dxlist_swap, dh0, dWx, dWh, db = rnn_backward(dhlist, hlist_cache)
        elif self.cell_type == 'lstm':
            dxlist_swap, dh0, dWx, dWh, db = lstm_backward(dhlist, hlist_cache)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)

        # (2)

        dxlist = np.swapaxes(dxlist_swap, 0, 1)
        dW_embed = word_embedding_backward(dxlist, xlist_cache)
        
        # (1)

        dfeatures, dW_proj, db_proj = fc_backward(dh0, h0_cache)

        for k, p in self.params.items():
            grads[k] = eval('d' + k)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.
        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.
        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.
        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.
        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned fc transformation to the next hidden state to     #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        ###########################################################################

        prev_h, _ = fc_forward(features, W_proj, b_proj) # (N, H)
        N, H = prev_h.shape
        prev_c, next_c = 0., 0.

        pred_captions = self._start * np.ones((N, 1), dtype=np.int32)
        captions[:, 0] = self._start

        for i in range(1, max_length):

            pred_x, _ = word_embedding_forward(pred_captions, W_embed) # (N, 1, W)
            pred_x = np.reshape(pred_x, (pred_x.shape[0], pred_x.shape[2]))

            if self.cell_type == 'rnn':
                next_h, _ = rnn_step_forward(pred_x, prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                next_h, next_c, _ = lstm_step_forward(pred_x, prev_h, prev_c, Wx, Wh, b)
            else:
                raise ValueError('Invalid cell_type "%s"' % self.cell_type)
            
            scores, _ = temporal_fc_forward(next_h.reshape(N, 1, H), W_vocab, b_vocab) #(N, 1, V)

            pred_captions = np.argmax(scores, axis=2)
            captions[:, i] = pred_captions.reshape(N,)

            prev_h = next_h
            prev_c = next_c


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
