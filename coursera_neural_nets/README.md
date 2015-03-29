

## Neural Networks for Machine Learning (Coursera)
First two assignments from the [Coursera Neural Networks class](https://www.coursera.org/course/neuralnets) converted from Matlab to Torch7.

### Installation
- [gfx.js](https://github.com/clementfarabet/gfx.js) for graphs.
- [matio](https://github.com/soumith/matio-ffi.torch) for loading matlab data files without having matlab installed. 
- [optim](https://github.com/torch/optim) optimization package.

Make sure gfx.js is running for running Assignment 1.

``` sh
 luajit -lgfx.start
```

Each assignment contains a starter_code folder where you can begin the assignment. 

Once complete you can compare against the solution folder.

### Assignment 1

Perceptron Algorithm.

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
