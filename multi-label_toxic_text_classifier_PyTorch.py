import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 42

pl.seed_everything(RANDOM_SEED)


def load_data(file_name, separator=';'):
    try:
        dataDF = pd.read_csv(file_name, sep=separator)
        print('FILE EXIST')
        return dataDF
    except IOError as ioe:
        # file didn't exist (or other issues)
        print('File do not exist!')
        print(ioe)
        return False


# Encapsulate the tokenization process in a PyTorch Dataset, and convert labels to tensors:
class ToxicCommentsDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.LABEL_COLUMNS = data.columns.tolist()[3:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        comment_text = data_row.comment_text
        labels = data_row[self.LABEL_COLUMNS]

        encoding = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            comment_text=comment_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )


# Encapsulating the custom dataset into a LightningDataModule:
class ToxicCommentDataModule(pl.LightningDataModule):

  def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128, num_workers=2):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
    self.num_workers = num_workers


  def setup(self, stage=None):
    self.train_dataset = ToxicCommentsDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )

    self.test_dataset = ToxicCommentsDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len
    )

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers
    )

  def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers
    )


class ToxicCommentTagger(pl.LightningModule):

    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


if __name__ == '__main__':
    data = load_data('data/train_balanced.csv', ';')

    train_df, val_df = train_test_split(data, test_size=0.05)
    train_df.shape, val_df.shape

    LABEL_COLUMNS = train_df.columns.tolist()[3:]

    BERT_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # Check the conversion of raw text into a list of tokens
    sample_row = data.sample(n=1).iloc[0]
    sample_comment = sample_row.comment_text
    sample_labels = sample_row[LABEL_COLUMNS]

    print(sample_comment)
    print(sample_labels.to_dict())

    encoding = tokenizer.encode_plus(
        sample_comment,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )
    encoding.keys()
    encoding['input_ids'].shape, encoding['attention_mask'].shape

    encoding['input_ids'].squeeze()[:20]
    encoding['attention_mask'].squeeze()[:20]

    # It is also possible to inverse the tokenization and get more or less back the words from the token ids:
    print(tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze())[:20])

    # Check the number of tokens per comment:
    token_counts = []
    for _, row in train_df.iterrows():
        token_count = len(tokenizer.encode(
            row["comment_text"],
            max_length=512,
            truncation=True
        ))
        token_counts.append(token_count)

    sns.histplot(token_counts)
    plt.xlim([0, 512]);
    plt.show()

    MAX_TOKEN_COUNT = 512

    # Check the dataset functionality and show some items
    train_dataset = ToxicCommentsDataset(
        train_df,
        tokenizer,
        max_token_len=MAX_TOKEN_COUNT
    )
    sample_item = train_dataset[0]
    sample_item.keys()

    sample_item["comment_text"]
    sample_item["labels"]
    sample_item["input_ids"].shape

    # Loading the BERT model
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)

    # sample_batch = next(iter(DataLoader(train_dataset, batch_size=8, num_workers=2)))
    # sample_batch["input_ids"].shape, sample_batch["attention_mask"].shape

    # ToxicCommentDataModule encapsulates all data loading logic and returns the necessary data loaders.
    N_EPOCHS = 5
    BATCH_SIZE = 12

    data_module = ToxicCommentDataModule(
        train_df,
        val_df,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_COUNT,
        num_workers=8
    )

    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS

    warmup_steps = total_training_steps // 5
    warmup_steps, total_training_steps

    # Creating an instance of the model
    model = ToxicCommentTagger(
        n_classes=len(LABEL_COLUMNS),
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="training/checkpoints",
        filename="training/best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name="toxic-comments")

    # Early stop of the loss doesn't improve for 10 epochs
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)
