from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Access the model configuration
config = model.config

# The number of layers can be found in the config
num_layers = config.num_hidden_layers

print(num_layers)
