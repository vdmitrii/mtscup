# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    # df_all['views'] = df_all['init_domain'].apply(change_m)
    data_idx['entropy'] = data_idx['url_host'].apply(entropy)
    data_idx['is_ip'] = data_idx['url_host'].apply(is_ip)
    data_idx['num_digits'] = data_idx['url_host'].apply(num_digits)
    data_idx['url_length'] = data_idx['url_host'].apply(url_length)
    data_idx['num_params'] = data_idx['url_host'].apply(num_params)
    data_idx['num_frag'] = data_idx['url_host'].apply(num_fragments)


if __name__ == '__main__':
    main()
