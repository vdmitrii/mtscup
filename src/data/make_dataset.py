# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

LOCAL_DATA_PATH = 'data/raw'
SPLIT_SEED = 42
DATA_FILE = 'competition_data_final_pqt'
TARGET_FILE = 'public_train.pqt'
SUBMISSION_FILE = 'submit_2.pqt'

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    id_to_submit = pd.read_parquet(input_filepath)
    df = pd.read_parquet(input_filepath, columns=['user_id', 'url_host', 'req_sum','req_max'])
    id_to_submit.to_csv(output_filepath, index=False)
    df.to_parquet(output_filepath, index=False)
    

if __name__ == '__main__':
    main()
