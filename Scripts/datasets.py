import pickle
import os.path
from typing import Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer

from Scripts import preprocess_data as prepro


class _PreBertTokenize:
    def __init__(self, pretrained_model_name_or_path):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)

    def tokenize(self, text, chunk_size=512):
        out_d = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')

        for name, token in tokens.items():
            token_chunks = token[0].split(chunk_size - 2)  # -2 to account for [cls] and [sep] tokens
            for chunk in token_chunks:
                if name == "input_ids":
                    # [CLS]/101 and seperator token [SEP]/102
                    chunk = torch.concat((torch.tensor([101]), chunk, torch.tensor([102])))
                elif name == "token_type_ids":
                    pass
                elif name == "attention_mask":
                    chunk = torch.concat((torch.ones(1), chunk, torch.ones(1)))
                else:
                    raise ValueError(f"unexpected name when tokenizing: {name}")

                chunk_len = chunk.size(0)
                if chunk_len < chunk_size:
                    pad_len = chunk_size - chunk_len
                    chunk = torch.concat((chunk, torch.zeros(pad_len)))

                out_d[name].append(chunk)

        out_d = {name: torch.stack(tokens) for name, tokens in out_d.items()}

        return {
            'input_ids': out_d['input_ids'].long(),
            'attention_mask': out_d['attention_mask'].int(),
            'token_type_ids': out_d['token_type_ids'].int()
        }


class FOMCImpactDataset(Dataset, _PreBertTokenize):
    """Beige Books and FOMC Impact on SP500 """
    def __init__(self, p_beige_books: str, p_fomc_impacts: str):
        """
        Dataset to return encoded beige book and impact of FOMC on sp500
        Args:
            p_beige_books: path to beige_books.csv
            p_fomc_impacts: path to fomc_impact.csv
        """
        Dataset.__init__(self)
        _PreBertTokenize.__init__(self, 'ProsusAI/finbert')

        df_bb = pd.read_csv(p_beige_books)
        df_bb.date = pd.to_datetime(df_bb.date)

        df_fomc = pd.read_csv(p_fomc_impacts)
        df_fomc.date = pd.to_datetime(df_fomc.date)

        df = prepro.merge_beige_books_impact(df_bb, df_fomc).reset_index()

        self.dates = df["impact_date"].sort_values(ascending=False).unique().tolist()
        self.df = df

    def __len__(self):
        """ Returns length of Dataset """
        return len(self.dates)

    def __getitem__(self, item: int) -> Tuple:
        """
        Get vectorizer encoded beige book, sp500 impact due to FOMC
        Args:
            item: index

        Returns:
            - vectorizer encoded beige book, one row for each district and one column for each word
            - impact FOMC had on sp500
        """
        date = self.dates[item]
        group = self.df.loc[self.df['impact_date'] == date]
        X = list(map(self.tokenize, group.sort_values('district').text))
        y = group.diff_norm.iloc[0]  # all diff_norm values in group are identical

        return X, torch.tensor(y)

if __name__ == "__main__":
    """ Example usage: """

    p_bb = "../Data/beige_books.csv"
    p_fomc = "../Data/fomc_impact.csv"

    dset = FOMCImpactDataset(p_bb, p_fomc)
    x, y = dset[2]

