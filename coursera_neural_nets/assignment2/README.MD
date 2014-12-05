
### Assignment 2
Neural network to predict next word in a sentence.

There are three solutions for this assignment as follows

1. Starter_code

   - Complete the sections in train.lua and fprop.lua where it says ‘Fill in code’.

2. Solution
 
   - The required code added to the starter code.

3. Solution_torch_nn

   - Uses the torch [nn](https://github.com/torch/nn) module.
   - Shows how the neural net can easily be created and using a lot less code.

#### Running the code

- Run main.lua using three words and the neural net will predict the fourth.

- All words must be from the vocab list of 250 words.

``` sh
 model = train(1);
 predict_next_word('life', 'in', 'new', model, 3);
```
The full list of words can be seen in the model.vocab_ByIndex  object.

Ouptut should be something similar to the following.

``` sh
 life in new york Prob: 0.953664
 life in new . Prob: 0.014226
 life in new world Prob: 0.005527
```

#### Neural Net Architecture
![network diagram](http://spark-public.s3.amazonaws.com/neuralnets/Programming%20Assignments/Assignment2/network.png)
