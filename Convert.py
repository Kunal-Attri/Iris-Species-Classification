import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import Bunch


def convert_to_dataframe(sk_data: Bunch) -> DataFrame:
    df = pd.DataFrame(data=np.c_[sk_data['data'], sk_data['target']])

    cols = [i.replace(" ", "").replace("(cm)", "") for i in sk_data['feature_names'] + ['target']]
    df.columns = cols

    col_map = dict()
    df['target'] = df['target'].convert_dtypes(convert_integer=True)
    for i, target in enumerate(sk_data['target_names']):
        col_map[i] = target
    df['target'] = df['target'].map(col_map)
    return df


def single_y_test_pred(y_test, y_pred) -> DataFrame:
    return pd.concat(
        [y_test.reset_index(), pd.DataFrame({"y_pred": y_pred})], axis=1
    )
