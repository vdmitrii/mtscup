# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    df = targets_data_idx.merge(usr_emb, how='inner', on=['user_id'])
    df = df[df['is_male'] != 'NA']
    df = df.dropna()
    df['is_male'] = df['is_male'].map(int)

    x_train, x_test, y_train, y_test = train_test_split(\
        df.drop(['age', 'domain', 'user_id', 'is_male', 'url_host'], axis=1), df['is_male'],\
                                                        test_size=0.2,\
                                                        random_state=SPLIT_SEED)

    # cat_fts = ['cpe_manufacturer_name', 'cpe_model_name', 'cpe_type_cd',\
    #            'cpe_model_os_type', 'category', 'postal_code']

    cat_fts = ['cpe_manufacturer_name',
        'cpe_model_name',
        'cpe_type_cd',
        'cpe_model_os_type',
        'category',
        'domain_cpe_manufacturer_name',
        'domain_cpe_model_name',
        'domain_cpe_type_cd',
        'domain_cpe_model_os_type',
        'domain_category',
        'cpe_manufacturer_name_cpe_model_name',
        'cpe_manufacturer_name_cpe_type_cd',
        'cpe_manufacturer_name_cpe_model_os_type',
        'cpe_manufacturer_name_category',
        'cpe_model_name_cpe_type_cd',
        'cpe_model_name_cpe_model_os_type',
        'cpe_model_name_category',
        'cpe_type_cd_cpe_model_os_type',
        'cpe_type_cd_category',
        'cpe_model_os_type_category'
        ]

    emb_fts = []

    train_pool = Pool(x_train,
                    y_train,
                    cat_features=cat_fts,
    #                   embedding_features=list(range(1, 61))
                    )
        
    test_pool = Pool(x_test,
                    y_test,
                    cat_features=cat_fts,
    #                   embedding_features=list(range(1, 61))
                    )

    clf = CatBoostClassifier(iterations=1_000,
                            task_type="GPU", 
                            devices='0:1',     
                            random_seed=SPLIT_SEED,
                            learning_rate=0.1,
    #                          l2_leaf_reg=10,
    #                          bagging_temperature=1,
                            early_stopping_rounds=20
                            )

    clf.fit(train_pool,
            eval_set=test_pool, 
            verbose=False, 
        )

    # clf = CatBoostClassifier(
    #         verbose=False, 
    #         iterations=1000,
    #         learning_rate=0.1,
    #         task_type="GPU",
    #         devices='0:1',
    #        )
    # clf.fit(x_train, y_train, verbose=False)

    sex_preds = clf.predict_proba(x_test)[:, 1]
    sex_gini = 2 * m.roc_auc_score(y_test, sex_preds) - 1

    print(f'GINI по полу {2 * m.roc_auc_score(y_test, sex_preds) - 1:2.3f}')

    clf.save_model('model_sex_eval')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
