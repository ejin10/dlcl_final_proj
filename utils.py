import evaluate 
from evaluate import evaluator
import numpy as np
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch
import os
import pickle as pkl

def load_data_from_dir(data_dir):
    dataset = load_from_disk(data_dir)
    dataset = dataset.remove_columns('text')
    dataset = dataset.rename_column('label', 'labels')
    dataset.set_format('torch')
    return dataset


def model_init(model_filepath):
    model = AutoModelForSequenceClassification.from_pretrained(model_filepath, trust_remote_code=True, num_labels=5)    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    model.to(device)
    return model


def pipeline_init(model_filepath):
    model = AutoModelForSequenceClassification.from_pretrained(model_filepath, trust_remote_code=True, num_labels=5, ignore_mismatched_sizes=True)    
    model.hi_transformer.embeddings = torch.load(os.path.join(model_filepath, 'embeds'))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    model.to(device)
    return model

def compute_metrics(eval_pred, average='macro'):
    "called at end of validation"
    print(f'evaluating with {average}')
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    accuracy_metric = evaluate.load('accuracy')
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')
    f1_metric = evaluate.load('f1')
    
    metrics = {'accuracy': accuracy_metric.compute(predictions=preds, references=labels)['accuracy'], 
            'precision': precision_metric.compute(predictions=preds, references=labels, average=average, zero_division=0)['precision'], 
            'recall': recall_metric.compute(predictions=preds, references=labels, average=average, zero_division=0)['recall'], 
            'f1': f1_metric.compute(predictions=preds, references=labels, average=average)['f1'], 
           }
    return metrics


def evaluate_model(dataset, model, model_name, average='macro', split='test'): 
    args = TrainingArguments(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=256,
        output_dir=os.path.join('evaluation', model_name),
        save_total_limit=3, 
        run_name=model_name,
        report_to=None
    )
          
    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=dataset['train'],
                      eval_dataset=dataset[split],
                      compute_metrics=compute_metrics,
                     )
    
    results = trainer.evaluate()  
    
    return results


def evaluate_models(dataset, model_dir, models_type, average='macro', split='test', save_to='results.pkl'):
    model_results = {}
    for model_name in os.listdir(model_dir): 
        print(f'evaluating {model_name}')
        model_filepath = os.path.join(model_dir, model_name)
        model = model_init(model_filepath) if models_type == 'models' else pipeline_init(model_filepath)
        model_results[model_name] = evaluate_model(dataset, model, model_name, average, split)
        with open(save_to, 'wb') as f:
            pkl.dump(model_results, f)
            
    return model_results