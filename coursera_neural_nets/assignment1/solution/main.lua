-- Coursera Neueral Nets for Machine Learning Assignment 1.
-- Perceptron Algorithm.

local matio = require 'matio'
require 'learn_perceptron'

-- Choose between dataset1 to dataset4.
dataset = 'data/dataset1.mat';

pos_examples_nobias = matio.load(dataset,'pos_examples_nobias')
neg_examples_nobias = matio.load(dataset,'neg_examples_nobias')
w_init = matio.load(dataset,'w_init')
w_gen_feas = matio.load(dataset,'w_gen_feas')

print(pos_examples_nobias);
print(neg_examples_nobias);

learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas)