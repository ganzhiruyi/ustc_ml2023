from typing import Optional, Dict
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments
from datasets import load_dataset
from transformers.utils import logging
import numpy as np
logger = logging.get_logger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the eval data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    prediction_file: str = field(default="prediction.csv")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        split_expr=":",
    ):
        super(SupervisedDataset, self).__init__()
        self.data = load_dataset('csv', data_files=data_path, 
            split= f'train[{split_expr}]')
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        # print('example:\n')
        # print(self.data[0], flush=True)
        # item = self.preprocessing(self.data[0])
        # print(item, flush=True)
        # print("input:", self.tokenizer.decode(item["input_ids"]), flush=True)
        # print("label:", self.tokenizer.decode(item["labels"]), flush=True)

    def __len__(self):
        return len(self.data)
        
    def preprocessing(self, example):
        out = self.tokenizer(example['fact'], max_length=self.model_max_length,padding=True, truncation=True)
        input_ids = out.input_ids
        attention_mask = out.attention_mask
        labels = example['label'] if 'label' in example else 0
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        return model_inputs

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

from transformers import MegatronBertForSequenceClassification,MegatronBertConfig
from transformers import BertTokenizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

   # Calculate precision, recall, and F1-score
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
import torch.distributed as dist
def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    config = MegatronBertConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = 34
    model = MegatronBertForSequenceClassification.from_pretrained(model_args.model_name_or_path,
        config = config
    )

    train_dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length,split_expr='1%:99%'
    )
    eval_dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length,split_expr="0%:1%"
    )
    predict_dataset = SupervisedDataset(
        data_args.eval_data_path, tokenizer, training_args.model_max_length,
        # split_expr="0%:1%"
    )
    collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer,    padding='longest', max_length=training_args.model_max_length)
    trainer = transformers.Trainer(
        model=model, args=training_args,
        compute_metrics=compute_metrics, 
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        tokenizer=tokenizer, data_collator=collator,
    )
    trainer.train()
    trainer.save_state()
    logger.info("*** Predict ***")
    predictions,labels, metrics = trainer.predict(predict_dataset)
    predictions = np.argmax(predictions, axis=1)
    pdata = load_dataset('csv', data_files=data_args.eval_data_path,
            split= 'train[:]')
    logger.info(predictions[0])
    if trainer.is_world_process_zero():
        logger.info(predictions.shape, labels.shape)
        pdata.remove_columns('fact').add_column('label', predictions).to_csv(training_args.prediction_file)
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()