require 'load_data'
require 'fprop'

-- This function trains a neural network language model.
function train(epochs)
--[[% Inputs:
%   epochs: Number of epochs to run.
% Output:
%   model: A struct containing the learned weights and biases and vocabulary.
]]--

start_time = os.clock();

-- SET HYPERPARAMETERS HERE.
batchsize = 100;  -- Mini-batch size.
learning_rate = 0.1;  -- Learning rate; default = 0.1.
momentum = 0.9;  -- Momentum; default = 0.9.
numhid1 = 50;  -- Dimensionality of embedding space; default = 50.
numhid2 = 200;  -- Number of units in hidden layer; default = 200.
init_wt = 0.01;  -- Standard deviation of the normal distribution
                 -- which is sampled to get the initial weights; default = 0.01

-- VARIABLES FOR TRACKING TRAINING PROGRESS.
show_training_CE_after = 100;
show_validation_CE_after = 1000;


-- LOAD DATA.
train_input, train_target, valid_input, valid_target, test_input, test_target, vocab, vocab_size, vocab_ByIndex = load_data(batchsize);

numwords = (#train_input)[1];
batchsize = (#train_input)[2];
numbatches = (#train_input)[3];

-- INITIALIZE WEIGHTS AND BIASES.
word_embedding_weights = torch.randn(vocab_size, numhid1);
embed_to_hid_weights = torch.randn(numwords * numhid1, numhid2) * init_wt ;
hid_to_output_weights = torch.randn(numhid2, vocab_size) * init_wt;
hid_bias = torch.zeros(numhid2, 1);
output_bias = torch.zeros(vocab_size, 1);

word_embedding_weights_delta = torch.zeros(vocab_size, numhid1);
word_embedding_weights_gradient = torch.zeros(vocab_size, numhid1);
embed_to_hid_weights_delta = torch.zeros(numwords * numhid1, numhid2);
hid_to_output_weights_delta = torch.zeros(numhid2, vocab_size);
hid_bias_delta = torch.zeros(numhid2, 1);
output_bias_delta = torch.zeros(vocab_size, 1);
expansion_matrix = torch.eye(vocab_size); -- Identity Matrix.
count = 0;
tiny = math.exp(-30);

-- TRAIN.
for epoch = 1,epochs do
  print('Epoch %d', epoch);
  this_chunk_CE = 0;
  trainset_CE = 0;
  -- LOOP OVER MINI-BATCHES.
  for m = 1, numbatches do
    input_batch = train_input[{ {}, {}, m}];
    target_batch = train_target[{ {}, {}, m}];

    -- FORWARD PROPAGATE.
    -- Compute the state of each layer in the network given the input batch
    -- and all weights and biases
    embedding_layer_state, hidden_layer_state, output_layer_state = fprop(input_batch,word_embedding_weights, 
      embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias);


    -- COMPUTE DERIVATIVE.
    -- Expand the target to a sparse 1-of-K vector.
    target_indexes =  torch.reshape(target_batch,(#target_batch)[2]):type('torch.LongTensor'); -- matrix to vector.
    expanded_target_batch = expansion_matrix:index(2, target_indexes);

    -- Compute derivative of cross-entropy loss function.
    error_deriv = output_layer_state - expanded_target_batch;
    
    -- MEASURE LOSS FUNCTION.
    s = torch.cmul(expanded_target_batch, torch.log(output_layer_state + tiny));
    CE = -torch.sum(s) / batchsize;
    count =  count + 1;
    this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count;
    trainset_CE = trainset_CE + (CE - trainset_CE) / m;
    print(string.format("Batch %.0f Train CE %f",m, this_chunk_CE));
    if math.fmod(m, show_training_CE_after) == 0 then
      print('\n');
      count = 0;
      this_chunk_CE = 0;
    end
      
    -- BACK PROPAGATE.
    -- OUTPUT LAYER.
    hid_to_output_weights_gradient =  hidden_layer_state * error_deriv:transpose(1,2);
    output_bias_gradient = torch.sum(error_deriv, 2);
    
    -- Derivative of siqmoid (logisitic activation function).
    s = torch.cmul((hid_to_output_weights * error_deriv), hidden_layer_state );
    back_propagated_deriv_1 = torch.cmul(s,(-hidden_layer_state+1));
    
    -- HIDDEN LAYER.
    -- FILL IN CODE. Replace the line below by one of the options.
       embed_to_hid_weights_gradient = torch.zeros(numhid1 * numwords, numhid2);
    -- Options:
    -- (a) embed_to_hid_weights_gradient = back_propagated_deriv_1:transpose(1,2) * embedding_layer_state;
    -- (b) embed_to_hid_weights_gradient = embedding_layer_state * back_propagated_deriv_1:transpose(1,2);
    -- (c) embed_to_hid_weights_gradient = back_propagated_deriv_1;
    -- (d) embed_to_hid_weights_gradient = embedding_layer_state;
  
    -- FILL IN CODE. Replace the line below by one of the options.
       hid_bias_gradient = torch.zeros(numhid2, 1);
    -- Options
    -- (a) hid_bias_gradient = torch.sum(back_propagated_deriv_1, 2);
    -- (b) hid_bias_gradient = torch.sum(back_propagated_deriv_1, 1);
    -- (c) hid_bias_gradient = back_propagated_deriv_1;
    -- (d) hid_bias_gradient = back_propagated_deriv_1:transpose(1,2);

    -- FILL IN CODE. Replace the line below by one of the options.
       back_propagated_deriv_2 = torch.zeros(numhid2, batchsize);
    -- Options
    -- (a) back_propagated_deriv_2 = embed_to_hid_weights * back_propagated_deriv_1;
    -- (b) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights;
    -- (c) back_propagated_deriv_2 = back_propagated_deriv_1:transpose(1,2) * embed_to_hid_weights;
    -- (d) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights:transpose(1,2);
    
    word_embedding_weights_gradient[{}] = 0;
    -- EMBEDDING LAYER.
    for w = 1, numwords do
       input_word_indexes =  input_batch[{w, {}}]:type('torch.LongTensor');
       em = expansion_matrix:index(2,input_word_indexes);
       
       b = back_propagated_deriv_2[{ {1 + (w - 1) * numhid1,w * numhid1},{} }];
       word_embedding_weights_gradient = word_embedding_weights_gradient + em * b:transpose(1,2);     
    end
    
    -- UPDATE WEIGHTS AND BIASES.
    word_embedding_weights_delta = torch.mul(word_embedding_weights_delta,momentum) + torch.div(word_embedding_weights_gradient,batchsize);
    word_embedding_weights = word_embedding_weights - torch.mul(word_embedding_weights_delta,learning_rate);
    
    
    embed_to_hid_weights_delta = torch.mul(embed_to_hid_weights_delta,momentum) + torch.div(embed_to_hid_weights_gradient,batchsize);
    embed_to_hid_weights = embed_to_hid_weights - torch.mul(embed_to_hid_weights_delta,learning_rate);
    
    hid_to_output_weights_delta = torch.mul(hid_to_output_weights_delta,momentum) + torch.div(hid_to_output_weights_gradient,batchsize);
    hid_to_output_weights = hid_to_output_weights - torch.mul(hid_to_output_weights_delta,learning_rate);
    
    hid_bias_delta = torch.mul(hid_bias_delta,momentum) + torch.div(hid_bias_gradient,batchsize);
    hid_bias = hid_bias -  torch.mul(hid_bias_delta,learning_rate);
      
    output_bias_delta = torch.mul(output_bias_delta,momentum) + torch.div(output_bias_gradient,batchsize);
    output_bias = output_bias - torch.mul(output_bias_delta,learning_rate);
    
    -- VALIDATE.
    if math.fmod(m, show_validation_CE_after) == 0 then
      print('\rRunning validation ...');
    
      embedding_layer_state, hidden_layer_state, output_layer_state = fprop(valid_input, word_embedding_weights, embed_to_hid_weights,
                    hid_to_output_weights, hid_bias, output_bias);
      datasetsize = (#valid_input)[2];
      
      target_indexes =  valid_target[{1, {}}]:type('torch.LongTensor');
      expanded_valid_target = expansion_matrix:index(2,target_indexes);
      
      s = torch.cmul(expanded_valid_target, torch.log(output_layer_state + tiny));
      CE = -torch.sum(s) / datasetsize;
      
      print(' Validation CE %.3f\n', CE);
    end
  end
  print(string.format('\rAverage Training CE %.3f\n', trainset_CE));
end
print('Finished Training.\n');
print(string.format('Final Training CE %.3f\n', trainset_CE));

-- EVALUATE ON VALIDATION SET.
print('\rRunning validation ...');

embedding_layer_state, hidden_layer_state, output_layer_state = fprop(valid_input, word_embedding_weights, embed_to_hid_weights,hid_to_output_weights, hid_bias, output_bias);
datasetsize = (#valid_input)[2];

t =  valid_target[{1, {}}];
expanded_valid_target = expansion_matrix:index(2,t:type('torch.LongTensor'));

s = torch.cmul(expanded_valid_target, torch.log(output_layer_state + tiny));
CE = -torch.sum(s) / datasetsize;
print(string.format('\rFinal Validation CE %.3f\n', CE));

-- EVALUATE ON TEST SET.
print('\rRunning test ...');
embedding_layer_state, hidden_layer_state, output_layer_state = fprop(test_input, word_embedding_weights, embed_to_hid_weights,
        hid_to_output_weights, hid_bias, output_bias);
datasetsize = (#test_input)[2];

t =  test_target[{1, {}}];
expanded_test_target = expansion_matrix:index(2,t:type('torch.LongTensor'));

s = torch.cmul(expanded_test_target, torch.log(output_layer_state + tiny));
CE = -torch.sum(s) / datasetsize;
print(string.format('\rFinal Test CE %.3f\n', CE));

local model = {};
model.word_embedding_weights = word_embedding_weights;
model.embed_to_hid_weights = embed_to_hid_weights;
model.hid_to_output_weights = hid_to_output_weights;
model.hid_bias = hid_bias;
model.output_bias = output_bias;
model.vocab = vocab;
model.vocab_ByIndex = vocab_ByIndex;

diff = os.clock() - start_time;
print(string.format('Training took %.2f seconds\n', diff));

return model;

end