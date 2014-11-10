
Run main.lua using three words and the neural net will predict the fourth.

All words must be from the vocab list of 250 words.

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