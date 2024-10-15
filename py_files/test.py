from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "vikhyatk/moondream2"
model_path = "./models/img2txt"

# Download the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path)
