require 'train'
require 'predict_next_word'

use_manual_technique = false;
epochs = 1;

-- Manual Training seems to require more epochs to get a similar error rate.
if use_manual_technique == true then epochs = 3; end

model = train(epochs,use_manual_technique);

predict_next_word('life', 'in', 'new', model, 3);
