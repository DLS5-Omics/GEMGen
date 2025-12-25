
import os
import sys
sys.path.append("/home/jiangqun/program/druggen/SFM_framework")

import pickle
import pandas as pd
import numpy as np
import lmdb
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sfm.data.prot_data.util import obj2bstr
from sfm.data.sci_data.NlmTokenizer import NlmLlama3Tokenizer

import warnings 
warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self, tokenizer_path, max_len=8192):
        self.IGNORE_INDEX = -100
        self.MAX_LEN = max_len
        self.NUM_WORKERS = cpu_count()
        print("加载 tokenizer ...")
        self.tokenizer = NlmLlama3Tokenizer.from_pretrained(tokenizer_path)

    def process_row(self, row_dict):
        try:
            # flag = 'Yes' if float(row_dict['score']) >= 0.5 else 'No'
            flag='Yes'
            cur_txt = f"{row_dict['description']} <mol>{row_dict['smiles']}</mol> {flag}"
            inputs_id = self.tokenizer.encode(cur_txt, add_special_tokens=True)
            if len(inputs_id) > self.MAX_LEN:
                return None  # 过滤超长序列
            origin_labels = [self.IGNORE_INDEX] * len(inputs_id)
            origin_labels[-2] = inputs_id[-2]
            origin_labels[-1] = inputs_id[-1]
            num_idx_list = np.zeros(len(inputs_id), dtype=int)
            num_idx_list[-2] = 1
            return [inputs_id, origin_labels, num_idx_list, float(row_dict['score'])]
        except Exception as e:
            print(f"[Error] Row skipped due to: {e}")
            return None

    def load_data(self, file_path, sep=","):
        print("读取数据中 ...")
        df = pd.read_csv(file_path, sep=sep)
        if 'score' not in df.columns:
            df['score'] = 1
        data_list = df.to_dict(orient='records')
        print(f"数据总量: {len(data_list)}")
        return data_list

    def preprocess_data(self, data_list):
        print(f"开始使用 {self.NUM_WORKERS} 核进行多进程预处理 ...")
        with Pool(processes=self.NUM_WORKERS) as pool:
            result_iter = pool.imap(self.process_row, data_list, chunksize=500)
            result_list = list(tqdm(result_iter, total=len(data_list)))
        # 清洗无效结果
        result_list = [item for item in result_list if item is not None]
        print(f"有效样本数: {len(result_list)}")
        return result_list

    def save_to_lmdb(self, result_list, save_path):
        print("保存至 LMDB ...")
        os.makedirs(save_path, exist_ok=True)
        env = lmdb.open(
            save_path,
            subdir=True,
            readonly=False,
            lock=False,
            readahead=False,
            map_size=(200 + 1) * 1024 ** 3,
        )
        keys = []
        with env.begin(write=True) as txn:
            for i, item in enumerate(result_list):
                data = pickle.dumps(item)
                txn.put(str(i).encode(), data)
                keys.append(i)

        metadata = {
            "keys": keys,
            "size": len(keys),
        }
        with env.begin(write=True) as txn:
            txn.put("metadata".encode(), obj2bstr(metadata))
        print("✅ LMDB 数据保存完成！")


def df2lmdb(data_path='data.tsv', save_path='data_lmdb'):
    # 设置路径
    path = "/home/jiangqun/program/druggen/SFM_framework/"
    os.chdir(path)
    print("当前工作目录已设置为：", os.getcwd())
    tokenizer_path = "/work/jiangqun/druggen/sfm/1b/llama/Meta-Llama-3-8B"
    processor = DataProcessor(tokenizer_path=tokenizer_path)
    data_list = processor.load_data(data_path, sep=',')
    result_list = processor.preprocess_data(data_list)
    processor.save_to_lmdb(result_list, save_path)


# df2lmdb(data_path=f'/work/jiangqun/druggen/gpt5/GPT5_valid_smiles_6cols.csv',
#         save_path=f'/work/jiangqun/druggen/gpt5/GPT5_valid_smiles_6cols_lmdb')
df2lmdb(data_path="/work/jiangqun/druggen/gpt5/K562_new/GPT5_valid_smiles_6cols.csv",
        save_path="/work/jiangqun/druggen/gpt5/K562_new/GPT5_valid_smiles_6cols_lmdb")

# for step in ['494', '988', '1482', '1976', '2470']:
#     df2lmdb(data_path=f'/work/jiangqun/druggen/sfm/results/K562_month10/gemgen_step{step}_valid_smiles_6cols.csv',
#         save_path=f'/work/jiangqun/druggen/sfm/results/K562_month10/gemgen_step{step}_valid_smiles_6cols_lmdb')

# df2lmdb(data_path=f'/work/jiangqun/druggen/sfm/results/K562_month10/K562_new_gt_with_random_smiles.csv',
#         save_path=f'/work/jiangqun/druggen/sfm/results/K562_month10/K562_new_gt_with_random_chembl_lmdb')