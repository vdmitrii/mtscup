# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    def tfidf_pca_emb(url_col, n=100, max_f=400_000):
        tfidf_vectorizer = TfidfVectorizer(max_features=None, analyzer='char',ngram_range=(2, 3), max_df=0.95, min_df=2) #, analyzer='char')  #, max_df=0.95, min_df=2, analyzer='char'
        url_col = url_col.apply(lambda x: ' '.join(x.split('.')))
        encoded_docs = tfidf_vectorizer.fit_transform(url_col)
        pca = TruncatedSVD(n_components=n)
        encoded_docs = pca.fit_transform(encoded_docs)
        pf = pd.DataFrame(encoded_docs) #.toarray()) without PCA
        pf.columns = [f'tf_{i}' for i in pf.columns]
        return pf
    # to_string = lambda x: ' '.join(x.split('.')
    # to_func = np.vectorize(to_string)
    # to_func(x)
    # res = to_string(x)

    tfidf_emb = tfidf_pca_emb(data_idx.url_host)
    data_idx = pd.concat([data_idx, tfidf_emb], axis=1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
