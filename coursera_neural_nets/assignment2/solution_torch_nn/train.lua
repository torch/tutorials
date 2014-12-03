require 'load_data'
require 'nn'
require 'optim'


function getNextBatch(train_input, train_target, m)
  
  input_batch = train_input[{ {}, {}, m }]:transpose(1,2);

  target_batch = train_target[{ {}, {}, m }]:transpose(1,2);
  target_batch = torch.reshape(target_batch,(#target_batch)[1]); -- convert targets to a vector;
  
  dataset = {};
  function dataset:size() return batchsize end 
  for i=1,dataset:size() do
      local input = input_batch[i];  
      local output = target_batch[i];
          
      dataset[i] = {input, output}
  end
  
  return dataset, input_batch, target_batch;
  
end

-- create closure to evaluate f(X) and df/dX
feval = function(x)
      -- get new parameters
      parameters:copy(x)

      -- reset gradients
      gradParameters:zero()

      -- f is the average of all criterions
      local f = 0

      -- evaluate function for complete mini batch
      for i = 1,(#inputs)[1] do
        -- estimate f
        local output = model:forward(inputs[i])
        local err = criterion:forward(output, targets[i])
        f = f + err

        -- estimate df/dW
        local df_do = criterion:backward(output, targets[i])
        model:backward(inputs[i], df_do) -- backprop.
      end

      -- normalize gradients and f(X)
      gradParameters:div((#inputs)[1])
      f = f/(#inputs)[1];

      print(f);
      -- return f and df/dX
      return f,gradParameters
                         
    end

-- This function trains a neural network language model.
function train(epochs,use_manual_technique)
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

-- VARIABLES FOR TRACKING TRAINING PROGRESS.
show_training_CE_after = 100;
show_validation_CE_after = 1000;


-- LOAD DATA.
train_input, train_target, valid_input, valid_target, test_input, test_target, vocab, vocab_size, vocab_ByIndex = load_data(batchsize);

numwords = (#train_input)[1];
batchsize = (#train_input)[2];
numbatches = (#train_input)[3];

model = nn.Sequential();
model:add( nn.LookupTable(vocab_size, numhid1)); -- lookuptable, so for 3 inputs will produce a 3 x 50 matrix.
model:add( nn.Reshape(numwords*numhid1));        -- reshape 3 x 50 matrix to 150 units which is the first layer.
model:add( nn.Linear(numwords*numhid1,numhid2)); -- second layer is 200 units.
model:add( nn.Sigmoid() );           
model:add( nn.Linear(numhid2,vocab_size) );      -- output layer is 250 units.
model:add( nn.LogSoftMax() ); 

criterion = nn.ClassNLLCriterion();
trainer = nn.StochasticGradient(model, criterion);
trainer.learningRate = learning_rate;
trainer.maxIteration = 1;

for epoch = 1,epochs do
  print(string.format('Epoch %.0f', epoch));
  
  for m = 1, numbatches do
    
    print(string.format("Batch #%.0f of %.0f ",m,numbatches));
   
    if use_manual_technique == false then
      
      dataset = getNextBatch(train_input, train_target, m);
      trainer:train(dataset); 
      
    else 
      -- Manual Training.
      _,inputs,targets = getNextBatch(train_input, train_target, m);
      
      optimState = { 
        learningRate = learning_rate,
        momentum = momentum
      };
      parameters,gradParameters = model:getParameters();
      
      -- call the stochastic gradient optimizer.
      optim.sgd(feval, parameters, optimState) 
    end

  end
end

print('Finished Training.\n');

--[[
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

]]--

diff = os.clock() - start_time;
print(string.format('Training took %.2f seconds\n', diff));

model.vocab = vocab;
model.vocab_ByIndex = vocab_ByIndex;

return model;

end