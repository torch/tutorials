require 'train'
require 'predict_next_word'
require 'display_nearest_words'

model = train(1);
predict_next_word('life', 'in', 'new', model, 3);
