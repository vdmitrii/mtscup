# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    targets = pd.read_parquet(f'{LOCAL_DATA_PATH}/{TARGET_FILE}')
    targets_data_idx = targets.merge(data_idx, how='inner', on=['user_id'])
    url_set = set(data['url_host'].values)
    print(f'{len(url_set)} urls')
    url_dict = {url: idurl for url, idurl in zip(url_set, range(len(url_set)))}
    usr_set = set(data['user_id'].values)
    print(f'{len(usr_set)} users')
    usr_dict = {usr: user_id for usr, user_id in zip(usr_set, range(len(usr_set)))}

    values = np.array(data['req_sum'])
    rows = np.array(data['user_id'].map(usr_dict))
    cols = np.array(data['url_host'].map(url_dict))

    mat = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))
    # mat = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))

    als = implicit.approximate_als.FaissAlternatingLeastSquares(factors=500, iterations=50, use_gpu=True, \
        calculate_training_loss=False, regularization=0.1)

    als.fit(mat)

    u_factors = als.model.user_factors
    d_factors = als.model.item_factors

    inv_usr_map = {v: k for k, v in usr_dict.items()}
    usr_emb = pd.DataFrame(u_factors.to_numpy())
    usr_emb.columns = [f'em_{i}' for i in usr_emb.columns]
    usr_emb['user_id'] = usr_emb.index.map(inv_usr_map)
    usr_emb.columns = usr_emb.columns.astype(str)
    usr_emb.to_parquet('user_emb.parquet', index=False)

    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
