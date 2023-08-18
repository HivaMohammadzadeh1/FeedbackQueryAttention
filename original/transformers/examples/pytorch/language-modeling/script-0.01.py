import torch
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2LMHeadModel

# Load the pre-trained GPT-2 model and tokenizer
#model_name = ./test-clm1/checkpoint-1500/  # You can also use "gpt2-medium" or "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained('./changed/')
model = GPT2Model.from_pretrained('./changed/', output_hidden_states=True)
#model = GPT2LMHeadModel.from_pretrained("./test-clm1/checkpoint-1500/", output_hidden_states=True)
#print(model)
# Step 3: Load and preprocess the Wikitext dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

text_data = dataset["train"]["text"][:1000]


# Encode each example sentence and store the results in a list
encoded_sentences = []
for sentence in text_data:
    encoded_sentence = tokenizer.encode_plus(sentence, return_tensors="pt", add_special_tokens=True)
    encoded_sentences.append(encoded_sentence)
    

# create separate lists for each layer
l2_norms = []
for i in range(11):
  l2_norms.append([])

for example_encoded in tqdm(encoded_sentences):
 if example_encoded['input_ids'].shape[1] > 0:
   with torch.no_grad():
     outputs = model(**example_encoded)
#     print(outputs)
     # Get the hidden states from the outputs
   hidden_states = outputs.hidden_states
   #print(hidden_states)
  # print(hidden_states[0].shape)
   difference = []
   for layer in range(len(hidden_states) - 2):
    difference.append(hidden_states[layer] - hidden_states[layer + 1])     
   for i in range(len(difference)):
     #print(difference[i].shape)
     for j in range(len(difference[i])):
        for k in range(len(difference[i][j])):
            l2_norms[i].append(torch.linalg.vector_norm(difference[i][j][k])) 
 
# computing average difference for each layer
average = []
for i in range(11):
  average.append(sum(l2_norms[i]) / len(l2_norms[i]))

      
#averages_sentences = [sum(element for element in layer_diffs_list) / len(layer_diffs_list) for layer_diffs_list in zip(*layer_diffs_list)]
#print(averages_sentences)
# Plot the average differences between consecutive layer embeddings
plt.figure(figsize=(10, 6))
plt.plot(range(len(average)), average, marker='o')
plt.xlabel('Layers')
plt.ylabel('Average L2 Norm Difference')
plt.title('Average L2 Norm Difference between Consecutive Layer Pairs for 1000 examples of wikitest')
plt.grid(True)
plt.savefig("figure.png")
plt.show()
