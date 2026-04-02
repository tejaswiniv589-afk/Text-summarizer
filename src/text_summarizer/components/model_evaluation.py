from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from evaluate import load as load_metric
import torch
import pandas as pd
from tqdm import tqdm
from text_summarizer.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        


    def generate_batch_sized_chunks(self,list_of_elements, batch_size):
        """Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self,dataset, metric, model, tokenizer,
                                batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu",
                                column_text="article",
                                column_summary="highlights"):
        # Convert generators to lists to avoid consuming them prematurely
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):

            # Tokenize inputs
            inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                            padding="max_length", return_tensors="pt")

            # Generate summaries
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                    attention_mask=inputs["attention_mask"].to(device),
                                    length_penalty=0.8, num_beams=8, max_length=128)

            # Decode generated tokens to text
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
                                for s in summaries]

            # Standardize formatting (optional but common for ROUGE)
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            # Add batch to the metric object
            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        # Compute and return final ROUGE scores
        score = metric.compute()
        return score
    
    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_matrix = load_metric("rouge") # Now this will work after pip install

        score = self.calculate_metric_on_test_ds(
            dataset=dataset_samsum_pt["test"][0:10],
            metric=rouge_matrix,
            model=model_pegasus,
            tokenizer=tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary"
        )
            # FIX: Access scores correctly based on the version of rouge library used
            # If using old datasets.load_metric, use score[rn].mid.fmeasure
            # If using new evaluate.load, use score[rn]
        rouge_dict = {}
        for rn in rouge_names:
            if hasattr(score[rn], 'mid'):
                rouge_dict[rn] = score[rn].mid.fmeasure
            else:
                rouge_dict[rn] = score[rn]

        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)
        print("Evaluation results saved to:", self.config.metric_file_name)
