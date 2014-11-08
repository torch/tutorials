function display_nearest_words(word, model, k)
-- Shows the k-nearest words to the query word.
-- Inputs:
--   word: The query word as a string.
--   model: Model returned by the training script.
--   k: The number of nearest words to display.
-- Example usage:
--   display_nearest_words('school', model, 10);

word_embedding_weights = model.word_embedding_weights;
vocab = model.vocab;
id = vocab[word];
if id == nil then
   print('Word not in vocabulary:', word);
  return;
end
-- Compute distance to every other word.
vocab_size = #model.vocab_ByIndex;
word_rep = word_embedding_weights[{id, {}}];
diff = word_embedding_weights - torch.repeatTensor(word_rep, vocab_size, 1);
distance = torch.sqrt(torch.sum(torch.cmul(diff,diff), 2));

-- Sort by distance.
d, order = torch.sort(distance);
d, order = torch.sort(distance:transpose(1,2));
--order = order(2:k+1);  -- The nearest word is the query word itself, skip that.
for i = 1,k do
  --fprintf('%s %.2f\n', vocab{order(i)}, distance(order(i)));
  print(string.format("%s %s", vocab_ByIndex[order[1][i]], distance[order[1][i]][1] ));

end

end