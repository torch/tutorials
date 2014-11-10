-- Coursera Neueral Nets for Machine Learning Assignment 1.
-- Perceptron Algorithm.

require 'torch'
require('plot_perceptron')
require 'matio'

-- Learns the weights of a perceptron and displays the results.
function learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas) 
 --[[
 Learns the weights of a perceptron for a 2-dimensional dataset and plots
 the perceptron at each iteration where an iteration is defined as one
 full pass through the data. If a generously feasible weight vector
 is provided then the visualization will also show the distance
 of the learned weight vectors to the generously feasible weight vector.
 Required Inputs:
   neg_examples_nobias - The num_neg_examples x 2 matrix for the examples with target 0.
       num_neg_examples is the number of examples for the negative class.
   pos_examples_nobias - The num_pos_examples x 2 matrix for the examples with target 1.
       num_pos_examples is the number of examples for the positive class.
   w_init - A 3-dimensional initial weight vector. The last element is the bias.
   w_gen_feas - A generously feasible weight vector.
 Returns:
   w - The learned weight vector.
]]
  print ("*** starting function:learn_perceptron");
  
-- Bookkeeping
num_neg_examples = (#neg_examples_nobias)[1]
num_pos_examples = (#pos_examples_nobias)[1]
num_err_history = {};
w_dist_history = {};

-- Here we add a column of ones to the examples in order to allow us to learn
-- bias parameters.
neg_examples = torch.Tensor(num_neg_examples,3):fill(1);
neg_examples[{ {},{2,3} }] = neg_examples_nobias

pos_examples = torch.Tensor(num_pos_examples,3):fill(1);
pos_examples[{ {},{2,3} }] = pos_examples_nobias

-- If weight vectors have not been provided, initialize them appropriately.
if (w_init==nil) then
    w = torch.rand(3,1)
else
    w = w_init;
end

if(w_gen_feas~=nil) then
    w_gen_feas = {};
end

-- Find the data points that the perceptron has incorrectly classified
-- and record the number of errors it makes.
iter = 1;
mistakes0, mistakes1 = eval_perceptron(neg_examples,pos_examples,w);
num_errs = #(mistakes0) + #(mistakes1);
num_err_history[0+1] = num_errs;
print('Iteration: ',iter);
print('# of errors: ',num_errs);
print();
print('weights:');
print(w);

-- If a generously feasible weight vector exists, record the distance
-- to it from the initial weight vector.
if (#(w_gen_feas) ~= 0) then
    w_dist_history[0+1]= norm(w - w_gen_feas);
end

plot_perceptron(neg_examples_nobias, pos_examples_nobias, mistakes0, mistakes1, num_err_history, w, w_dist_history);
   

-- Iterate until the perceptron has correctly classified all points.
while (num_errs > 0) do
    iter = iter + 1;

    -- Update the weights of the perceptron.
    w = update_weights(neg_examples,pos_examples,w);

    -- If a generously feasible weight vector exists, record the distance
    -- to it from the current weight vector.
    if (#(w_gen_feas) ~= 0) then
       w_dist_history[0+1] = norm(w - w_gen_feas);
    end

    -- Find the data points that the perceptron has incorrectly classified.
    -- and record the number of errors it makes.
    mistakes0, mistakes1 = eval_perceptron(neg_examples,pos_examples,w); 
    num_errs = #(mistakes0) + #(mistakes1);
    num_err_history[0+1] = num_errs;

    print('Iteration: ',iter);
    print('# of errors: ',num_errs);
    print();
    print('weights:');
    print(w);

end

end


--WRITE THE CODE TO COMPLETE THIS FUNCTION
function update_weights(neg_examples, pos_examples, w_current)
--[[
% Updates the weights of the perceptron for incorrectly classified points
% using the perceptron update algorithm. This function makes one sweep
% over the dataset.
% Inputs:
%   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
%       num_neg_examples is the number of examples for the negative class.
%   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
%       num_pos_examples is the number of examples for the positive class.
%   w_current - A 3-dimensional weight vector, the last element is the bias.
% Returns:
%   w - The weight vector after one pass through the dataset using the perceptron
%       learning rule.
%%]]
w = w_current;
alpha = 1;
pos_class = 1;
neg_class = -1;

num_neg_examples = (#neg_examples)[1];
num_pos_examples = (#pos_examples)[1];
for i=1, num_neg_examples do
   
    this_case = torch.Tensor(1,3);
    this_case[{ {1},{} }]  = neg_examples[{ i,{} }];
    
    activation = this_case*w;
    if (activation[{ 1, 1}] >= 0) then
        --YOUR CODE HERE
        t = neg_class;
        --  
        -- Perceptron Algorithm.
        --   (T+1)     (T)
        -- w       = w     a(Phi * Target)
        --
        w = w + (this_case * t)*alpha;
    end
end
for i=1, num_pos_examples do
    this_case = torch.Tensor(1,3);
    this_case[{ {1},{} }]  = pos_examples[{ i,{} }];

    activation = this_case*w;
    if (activation[{ 1, 1}] < 0) then
        --YOUR CODE HERE
        t = pos_class;
         --  
        -- Perceptron Algorithm.
        --   (T+1)     (T)
        -- w       = w     a(Phi * Target)
        --
        w = w + (this_case * t)*alpha;
    end
end

    return w;

end

function eval_perceptron(neg_examples, pos_examples, w)
--[[
% Evaluates the perceptron using a given weight vector. Here, evaluation
% refers to finding the data points that the perceptron incorrectly classifies.
% Inputs:
%   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
%       num_neg_examples is the number of examples for the negative class.
%   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
%       num_pos_examples is the number of examples for the positive class.
%   w - A 3-dimensional weight vector, the last element is the bias.
% Returns:
%   mistakes0 - A vector containing the indices of the negative examples that have been
%       incorrectly classified as positive.
%   mistakes0 - A vector containing the indices of the positive examples that have been
%       incorrectly classified as negative.
%%]]

num_neg_examples = (#neg_examples)[1];
num_pos_examples = (#pos_examples)[1];
mistakes0 = {};
mistakes1 = {};

for i=1, (num_neg_examples) do

    x = torch.Tensor(1,3);
    x[{ {1},{} }] = neg_examples[{ i,{} }];
   
    activation = x * w
    if (activation[{ 1, 1}]  >= 0) then
        mistakes0[#mistakes0+1] = i
    end
end
for i=1, (num_pos_examples) do
   
    x = torch.Tensor(1,3);
    x[{ {1},{} }]  = pos_examples[{ i,{} }];
   
    activation = x * w
    if (activation[{ 1, 1}] < 0) then
        mistakes1[#mistakes1+1] = i
    end
end

return mistakes0, mistakes1

end