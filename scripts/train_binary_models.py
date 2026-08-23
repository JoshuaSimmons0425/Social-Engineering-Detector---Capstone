import gc
import sys
import torch 
from torch.nn import BCELoss
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
    baseline_model.run_pipeline()

    # Train the Bert model and evaluate it

    training_data = TextDataset(training_set, mode='answerdotai/ModernBERT-base', max_len=512)
    fitted_encoder = training_data.preprocess_labels(training=True)

    validation_data = TextDataset(validation_set, mode='answerdotai/ModernBERT-base', max_len=512, encoder=fitted_encoder)
    validation_data.preprocess_labels(training=False)

    training_loader = DataLoader(training_data, batch_size=4, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=4, shuffle=False, num_workers=0)

    epochs = 5
    optimizer = "adamw"
    learning_rate = 2e-5
    bert_model = BERTClassifier(
        n_classes=2,
        train_loader=training_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        criterion=BCELoss(),
        epochs=epochs,
        learning_rate=learning_rate,
        pretrained_model_name='answerdotai/ModernBERT-base'
    )
    bert_model.run_pipeline(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Save the trained models and their evaluation metrics

    baseline_model.save_model(output_dir='models/binary/baseline/uncalibrated')
    baseline_model.save_metrics(output_dir='experiments/binary/baseline/uncalibrated', output_format='txt')
    bert_model.save_model(path='models/binary/bert/uncalibrated/uncalibrated_bert_model.pt')
    bert_model.save_metrics(path='experiments/binary/bert/uncalibrated/uncalibrated_bert_model_metrics.txt', output_format='txt')

    sys.exit(0)

if __name__ == "__main__":
    main()