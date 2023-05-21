import pandas as pd
import click


@click.command()
@click.argument("query_string", type=click.STRING)
@click.argument("output_path", type=click.Path())
def get_data(query_string: str, output_path: str):
    # !wget 'https://storage.yandexcloud.net/ds-ods/files/materials/124f46f0/competition_data_final_pqt.zip' -O raw/competition_data_final_pqt.zip
    # !wget 'https://storage.yandexcloud.net/ds-ods/files/materials/adfd2b94/submit_2.pqt' -O raw/submit_2.pqt
    # !wget 'https://storage.yandexcloud.net/ds-ods/files/materials/f2fadc4d/public_train.pqt' -O raw/public_train.pqt

    # !unzip raw/competition_data_final_pqt.zip
    # !rm -rf raw/competition_data_final_pqt.zip

    df = pd.DataFrame(r.json()['elements'])
    df.to_csv(output_path)


if __name__ == "__main__":
    get_data()
