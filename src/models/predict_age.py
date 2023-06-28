# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


DEVICE = 'GPU'
ITERATIONS = 1_000
SPLIT_SEED = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.1
COLUMNS_TO_DROP = ['age', 'domain', 'user_id', 'is_male', 'url_host']

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    df = targets_data_idx.merge(usr_emb, how='inner', on=['user_id'])
    df = df[df['is_male'] != 'NA']
    df = df.dropna()
    df['is_male'] = df['is_male'].map(int)

    x_train, x_test, y_train, y_test = train_test_split(\
        df.drop(COLUMNS_TO_DROP, axis=1), df['is_male'],\
                test_size=TEST_SIZE,\
                random_state=SPLIT_SEED)

    train_pool = Pool(x_train,
                    y_train,
                    cat_features=['category']
                    )
        
    test_pool = Pool(x_test,
                    y_test,
                    cat_features=['category']
                    )

    clf = CatBoostClassifier(iterations=ITERATIONS,
                            task_type=DEVICE,    
                            random_seed=SPLIT_SEED,
                            learning_rate=LEARNING_RATE,
                            early_stopping_rounds=20
                            )

    clf.fit(train_pool,
            eval_set=test_pool, 
            verbose=False, 
        )

    sex_preds = clf.predict_proba(x_test)[:, 1]
    sex_gini = 2 * m.roc_auc_score(y_test, sex_preds) - 1

    # print(f'GINI по полу {2 * m.roc_auc_score(y_test, sex_preds) - 1:2.3f}')

    clf.save_model('model_sex_eval')


if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
