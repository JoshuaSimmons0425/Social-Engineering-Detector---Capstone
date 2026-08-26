import gc
import sys
import torch 
import yaml
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import pandas as pd
from src.datasets.textdatasets import TextDataset
from src.modelling.binary.bert.bert_model import BERTClassifier
from src.modelling.binary.baseline.baseline_model import TFIDFBaselineModel

def main():

    # Clear GPU memory before starting the training process
    gc.collect()
    torch.cuda.empty_cache()

    with open('data/splits/training_50_50.csv', 'r', encoding='utf-8', errors='replace') as f:
        training_set = pd.read_csv(f)
    with open('data/splits/validation_50_50.csv', 'r', encoding='utf-8', errors='replace') as f:
        validation_set = pd.read_csv(f)

    text_column = 'Full_Text'
    label_column = 'Label'

    # Train and evaluate the baseline model

    baseline_model = TFIDFBaselineModel(training_set, validation_set, text_column=text_column, label_column=label_column)

    print("Running the pipeline for the baseline model...")
    baseline_model.run_pipeline()

    # Train the Bert model and evaluate it

    with open('config/binary_model.yaml', 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['data']['batch_size']
    tokenizer_name = config['data']['tokenizer_name']
    max_len = config['data']['max_length']
    

    training_data = TextDataset(training_set, mode=tokenizer_name, max_len=max_len)
    fitted_encoder = training_data.preprocess_labels(training=True)

    validation_data = TextDataset(validation_set, mode=tokenizer_name, max_len=max_len, encoder=fitted_encoder)
    validation_data.preprocess_labels(training=False)

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    learning_rate = float(config['bert_model']['learning_rate'])
    n_classes = config['bert_model']['n_classes']
    epochs = config['bert_model']['epochs']
    optimizer = config['bert_model']['optimizer']
    model_name = config['bert_model']['model_name']
        

    bert_model = BERTClassifier(
        n_classes=n_classes,
        train_loader=training_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        epochs=epochs,
        learning_rate=learning_rate,
        pretrained_model_name=model_name
    )

    print("Running the pipeline for the BERT model...")
    bert_model.run_pipeline(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Save the trained models and their evaluation metrics
    baseline_model.save_model(output_dir='models/binary/baseline/uncalibrated')
    baseline_model.save_metrics(output_dir='experiments/binary/baseline/uncalibrated', output_format='txt')
    bert_model.save_model(path='models/binary/bert/uncalibrated/uncalibrated_bert_model.pt')
    bert_model.save_metrics(path='experiments/binary/bert/uncalibrated/uncalibrated_bert_model_metrics.txt', output_format='txt')

    # Save the fitted label encoder for future use
    training_data.save_encoder(output_dir='models/binary/bert/uncalibrated')

    sys.exit(0)

if __name__ == "__main__":
    main()