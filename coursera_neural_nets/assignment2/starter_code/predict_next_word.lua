

function predict_next_word(word1, word2, word3, model, k)
-- Predicts the next word.
-- Inputs:
--   word1: The first word as a string.
--   word2: The second word as a string.
--   word3: The third word as a string.
--   model: Model returned by the training script.
--   k: The k most probable predictions are shown.
-- Example usage:
--   predict_next_word('john', 'might', 'be', model, 3);
--   predict_next_word('life', 'in', 'new', model, 3);

word_embedding_weights = model.word_embedding_weights;
vocab = model.vocab;
id1 = vocab[word1];
id2 = vocab[word2];
id3 = vocab[word3];
if id1 == nil then
   print('Word not in vocabulary:', word1);
  return;
end
if id2 == nil then
   print('Word not in vocabulary:', word2);
  return;
end
if id3 == nil then
   print('Word not in vocabulary:', word3);
  return;
end
--input = [id1; id2; id3];
input = torch.Tensor({id1,id2,id3}):reshape(3,1);

embedding_layer_state, hidden_layer_state, output_layer_state = fprop(input, model.word_embedding_weights, model.embed_to_hid_weights,
  model.hid_to_output_weights, model.hid_bias, model.output_bias);

prob, indices = torch.sort(output_layer_state:transpose(1,2),true);
vocab_ByIndex = model.vocab_ByIndex;

for i = 1,k do
  print(string.format("%s %s %s %s Prob: %f", word1, word2, word3, vocab_ByIndex[indices[1][i]], prob[1][i]));
end


end