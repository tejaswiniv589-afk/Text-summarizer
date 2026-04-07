import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
from text_summarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        print("Loading model... (this may take a minute)")
        self.pipe = pipeline(
            "summarization",
            model=self.config.model_path,
            tokenizer=self.tokenizer
        )
        print("Model loaded successfully!")

    def predict(self, text, summary_length: int = None):
        # Truncate input to prevent hanging on long texts (e.g., limit to 512 tokens)
        tokens = self.tokenizer.encode(text, truncation=True, max_length=512)
        truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)

        # If the user provided a target length, use it; otherwise use a safe dynamic default.
        if summary_length:
            max_len = max(10, min(128, summary_length))
            min_len = max(10, min(max_len, int(max_len * 0.7)))
        else:
            input_length = len(truncated_text.split())
            max_len = max(30, min(128, input_length // 2))
            min_len = 10

        gen_kwargs = {
            "max_length": max_len,
            "min_length": min_len,
            "num_beams": 4,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        }

        print("Dialogue:")
        print(truncated_text)
        print(">>> Starting inference...")

        output = self.pipe(truncated_text, **gen_kwargs)[0]["summary_text"]

        print(">>> Inference complete!")

        # Clean up output
        output = output.replace("<n>", " ")
        output = re.sub(r'\s+([.,])', r'\1', output)
        output = re.sub(r'\.,', ',', output)
        output = re.sub(r'\.{2,}', '.', output)
        output = re.sub(r'\s+', ' ', output)
        output = output.strip()

        print("\nModel Summary:")
        print(output)

        return output