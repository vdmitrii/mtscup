# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    uniq_urls = data.url_host.unique()
    regdom_dict = {}
    domain_dict = {}

    for idx, url in enumerate(uniq_urls):
        ext = tldextract.extract(url)
        domain_dict[url] = ext.domain 
        regdom_dict[url] = ext.registered_domain
        # [ext.subdomain, ext.domain, ext.suffix, ext.registered_domain, url]
        
    data['domain'] = data.url_host.map(regdom_dict)
    data['domain'] = data.domain.astype('category')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
