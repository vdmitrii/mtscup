# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    pass


    def change_m(line):
        if 'K'in line:
            tmp = 1000
        elif 'M' in line:
            tmp = 1000000
        line = line.split('.')[0]
        res = int(line) * tmp
        return res


    def entropy(url):
        string = url.replace('NotGiven', '')
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy


    def is_ip(url):
        string = url
        flag = False
        if ("." in string):
            elements_array = string.strip().split(".")
            if(len(elements_array) == 4):
                for i in elements_array:
                    if (i.isnumeric() and int(i)>=0 and int(i)<=255):
                        flag=True
                    else:
                        flag=False
        if flag:
            return 1 
        else:
            return 0


    def num_digits(url):
        digits = [i for i in url if i.isdigit()]
        return len(digits)


    def url_length(url):
        return len(url)


    def num_params(url):
        params = url.split('.')
        return len(params) - 1


    def num_fragments(url):
        fragments = url.split('-')
        return len(fragments)


    print('Add url feats 2')
    # df_all['views'] = df_all['init_domain'].apply(change_m)
    data_idx['entropy'] = data_idx['url_host'].apply(entropy)
    data_idx['is_ip'] = data_idx['url_host'].apply(is_ip)
    data_idx['num_digits'] = data_idx['url_host'].apply(num_digits)
    data_idx['url_length'] = data_idx['url_host'].apply(url_length)
    data_idx['num_params'] = data_idx['url_host'].apply(num_params)
    data_idx['num_frag'] = data_idx['url_host'].apply(num_fragments)


    print('Add url feats 3')
    similarweb = pd.read_csv('/kaggle/input/sim-web/similarweb.csv')
    records = similarweb.to_dict('records')
    # {'domain': 'national-lottery.co.uk', 'rank': '867', 
    # 'country': 'United Kingdom', 'country_rank': '37', 
    # 'category': 'gambling/lottery', 'categ_rank': '1', 
    # 'month_views': '36.84M', 'search_traf': '35.4%'}
    sim_dict = {}
    for r in records:
        sim_dict[r['domain']] = r['category']

    data_idx['category'] = data_idx.domain.map(sim_dict)
    data_idx['category'] =  data_idx.category.astype('category')
    data_idx['category'] =  data_idx.category.cat.add_categories("NoCategory").fillna("NoCategory")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
