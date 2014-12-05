
### Assignment 2 (using the torch nn module)
Neural network to predict next word in a sentence.

This solution shows how the same network can easily be created using the torch [nn](https://github.com/torch/nn) module.

The neural net is instead created using just a few lines of code.

``` sh
 -- Create the neural net.
 model = nn.Sequential();
 model:add( nn.LookupTable(vocab_size, numhid1)); -- lookuptable,3 inputs will output a 3x50 matrix.
 model:add( nn.Reshape(numwords*numhid1));        -- reshape 3 x 50 matrix to 150 units.
 model:add( nn.Linear(numwords*numhid1,numhid2)); -- second layer is 200 units.
 model:add( nn.Sigmoid() );                       -- activation function.
 model:add( nn.Linear(numhid2,vocab_size) );      -- output layer is 250 units.
 model:add( nn.LogSoftMax() ); 
```

Create the trainer and set the loss function.


``` sh
 -- Minimize the negative log-likelihood
 criterion = nn.ClassNLLCriterion();
 trainer = nn.StochasticGradient(model, criterion);
 trainer.learningRate = learning_rate;
 trainer.maxIteration = 1; 
```

Train the model.

``` sh
 dataset = getNextBatch(train_input, train_target, m);
 trainer:train(dataset); 
```

The code also shows how to train the neural network manually.