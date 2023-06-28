import pandas as pd
import click


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(query_string: str, output_path: str):
    # !wget 'https://storage.yandexcloud.net/ds-ods/files/materials/124f46f0/competition_data_final_pqt.zip' -O raw/competition_data_final_pqt.zip
    # !wget 'https://storage.yandexcloud.net/ds-ods/files/materials/adfd2b94/submit_2.pqt' -O raw/submit_2.pqt
    # !wget 'https://storage.yandexcloud.net/ds-ods/files/materials/f2fadc4d/public_train.pqt' -O raw/public_train.pqt
    # !unzip raw/competition_data_final_pqt.zip
    # !rm -rf raw/competition_data_final_pqt.zip

    df = pd.read_parquet(input_filepath)
    df.to_csv(output_filepath)


if __name__ == "__main__":
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)
    # load_dotenv(find_dotenv())
    main()
