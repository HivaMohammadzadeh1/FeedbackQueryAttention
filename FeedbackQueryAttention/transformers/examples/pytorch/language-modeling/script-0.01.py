import torch
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
from datasets import load_dataset

# Load the pre-trained GPT-2 model and tokenizer
model_name = "test-clm"  # You can also use "gpt2-medium" or "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)

# Step 3: Load and preprocess the Wikitext dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

text_data = dataset["train"]["text"][:1000]


# Encode each example sentence and store the results in a list
encoded_sentences = []
for sentence in text_data:
    encoded_sentence = tokenizer.encode_plus(sentence, return_tensors="pt", add_special_tokens=True)
    encoded_sentences.append(encoded_sentence)
    

layer_diffs_list = []
for example_encoded in tqdm(encoded_sentences):
  if example_encoded['input_ids'].shape[1] > 0:
      with torch.no_grad():
          outputs = model(**example_encoded)
       # Get the hidden states from the outputs
      hidden_states = outputs.hidden_states
      #print(len(hidden_states))
      difference = []
      for layer in range(len(hidden_states) - 2):
        difference.append(hidden_states[layer] - hidden_states[layer + 1]) 

      l2_norms = []
      for i in range(len(difference)):
        l2_norms_per_token = []
        for j in range(len(difference[i])):
          l2_norms_per_token.append(torch.norm(difference[i][j], dim=-1).mean()) 
        l2_norms.append(l2_norms_per_token)
        averages = []
        for lst in l2_norms:
            average = sum(lst) / len(lst)
            averages.append(average)

      layer_diffs_list.append(averages)
      
averages_sentences = [sum(element for element in layer_diffs_list) / len(layer_diffs_list) for layer_diffs_list in zip(*layer_diffs_list)]
print(averages_sentences)
# Plot the average differences between consecutive layer embeddings
plt.figure(figsize=(10, 6))
plt.plot(range(len(averages_sentences)), averages_sentences, marker='o')
plt.xlabel('Layers')
plt.ylabel('Average L2 Norm Difference')
plt.title('Average L2 Norm Difference between Consecutive Layer Pairs for 1000 examples of wikitest')
plt.grid(True)
plt.savefig("figure3.png")
plt.show()



