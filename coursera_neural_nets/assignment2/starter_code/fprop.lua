
require 'nn'

function fprop(input_batch, word_embedding_weights, embed_to_hid_weights, 
                  hid_to_output_weights, hid_bias, output_bias)
--[[% This method forward propagates through a neural network.
% Inputs:
%   input_batch: The input data as a matrix of size numwords X batchsize where,
%     numwords is the number of words, batchsize is the number of data points.
%     So, if input_batch(i, j) = k then the ith word in data point j is word
%     index k of the vocabulary.
%
%   word_embedding_weights: Word embedding as a matrix of size
%     vocab_size X numhid1, where vocab_size is the size of the vocabulary
%     numhid1 is the dimensionality of the embedding space.
%
%   embed_to_hid_weights: Weights between the word embedding layer and hidden
%     layer as a matrix of soze numhid1*numwords X numhid2, numhid2 is the
%     number of hidden units.
%
%   hid_to_output_weights: Weights between the hidden layer and output softmax
%               unit as a matrix of size numhid2 X vocab_size
%
%   hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.
%
%   output_bias: Bias of the output layer as a matrix of size vocab_size X 1.
%
% Outputs:
%   embedding_layer_state: State of units in the embedding layer as a matrix of
%     size numhid1*numwords X batchsize
%
%   hidden_layer_state: State of units in the hidden layer as a matrix of size
%     numhid2 X batchsize
%
%   output_layer_state: State of units in the output layer as a matrix of size
%     vocab_size X batchsize
%]]--

numwords = input_batch:size(1);
batchsize = input_batch:size(2);
vocab_size = (#word_embedding_weights)[1];
numhid1 = (#word_embedding_weights)[2]
numhid2 = (#embed_to_hid_weights)[2];

-- COMPUTE STATE OF WORD EMBEDDING LAYER.
-- Look up the inputs word indices in the word_embedding_weights matrix.
-- embedding_layer_state is the 1st layer in the neural net.
-- With 3 input words and 50 nodes in the 1st layer there will be 150 weights.
-- This results in a matrix of numhid1 * numwords, batchsize.
-- i.e 150 x 100.
-- No calculations for this layer. Just lookup the weight for each input word for first layer.
input = torch.reshape(input_batch, 1, numwords * batchsize);
input_word_indexes = torch.reshape(input,(#input)[2]):type('torch.LongTensor'); -- converts matrix to a vector.
word_weights = word_embedding_weights:index(1,input_word_indexes);
embedding_layer_state = torch.reshape(word_weights:transpose(1,2) ,numhid1 * numwords, batchsize);

-- COMPUTE STATE OF HIDDEN LAYER.
-- Compute inputs to hidden units.
-- Multipy 1st layer weights by 2nd layer weights
inputs_to_hidden_units = embed_to_hid_weights:transpose(1,2) * embedding_layer_state + torch.repeatTensor(hid_bias, 1, batchsize);

-- Apply logistic activation function.
-- FILL IN CODE. Replace the line below by one of the options.
   hidden_layer_state = torch.zeros(numhid2, batchsize);
-- Options
--[[
  (a) hidden_layer_state = ( ((inputs_to_hidden_units):exp():add(1)):pow(-1) );
  (b) hidden_layer_state = ( ((inputs_to_hidden_units):exp():sub(1)):pow(-1) );
  (c) hidden_layer_state = ( ((-inputs_to_hidden_units):exp():add(1)):pow(-1) );
  (d) hidden_layer_state = -( ((-inputs_to_hidden_units):exp():add(1)):pow(-1) );
]]--

-- COMPUTE STATE OF OUTPUT LAYER.
-- Compute inputs to softmax.
-- FILL IN CODE. Replace the line below by one of the options.
   inputs_to_softmax = torch.rand(vocab_size, batchsize);
-- Options
--[[
  (a) inputs_to_softmax = hid_to_output_weights:transpose(1,2) * hidden_layer_state +  torch.repeatTensor(output_bias, 1, batchsize);
  (b) inputs_to_softmax = hid_to_output_weights:transpose(1,2) * hidden_layer_state +  torch.repeatTensor(output_bias, batchsize,1);
  (c) inputs_to_softmax = hidden_layer_state * hid_to_output_weights:transpose(1,2) +  torch.repeatTensor(output_bias, 1, batchsize);
  (d) inputs_to_softmax = hid_to_output_weights * hidden_layer_state +  torch.repeatTensor(output_bias, batchsize,1);
--]]

-- Subtract maximum. 
-- Remember that adding or subtracting the same constant from each input to a
-- softmax unit does not affect the outputs. Here we are subtracting maximum to
-- make all inputs <= 0. This prevents overflows when computing their
-- exponents. 
inputs_to_softmax = inputs_to_softmax - torch.repeatTensor(torch.max(inputs_to_softmax,1), vocab_size, 1);

-- Compute exp.
output_layer_state = torch.exp(inputs_to_softmax);

-- Normalize to get probability distribution.
ols_sum = torch.repeatTensor(torch.sum(output_layer_state, 1), vocab_size, 1);
output_layer_state = torch.cdiv(output_layer_state,ols_sum);

return embedding_layer_state, hidden_layer_state, output_layer_state;

end