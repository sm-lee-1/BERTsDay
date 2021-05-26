# -*- coding: utf-8 -*-

# 필요한 모듈 불러오기

import argparse
import os
import pickle
import tensorflow as tf

from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# read command-line parameters
parser = argparse.ArgumentParser('Evaluating the BERT / ALBERT NLU model')
parser.add_argument('--model', '-m', help = 'Path to BERT / ALBERT NLU model', type = str, required = True)
parser.add_argument('--type', '-tp', help = 'bert or albert', type = str, default = 'bert', required = False)
parser.add_argument('--bertpath', '-bp', help = '프리트레인된 BERT 모듈 경로', type = str, default = "/content/drive/MyDrive/bert-module")



VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
load_folder_path = args.model
type_ = args.type

# this line is to enable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto(intra_op_parallelism_threads=0,
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        device_count = {'GPU': 1})
sess = tf.compat.v1.Session(config=config)

if type_ == 'bert':
    bert_model_hub_path = args.bertpath
    is_bert = True
elif type_ == 'albert':
    bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'
    is_bert = False
else:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))


# 모델과 벡터라이저 불러오기
vocab_file = os.path.join(bert_model_hub_path, "assets/vocab.korean.rawtext.list")
bert_to_array = BERTToArray(is_bert, vocab_file)

#모델
print('Loading models ...')
if not os.path.exists(load_folder_path):
    raise FileNotFoundError('Folder `%s` not exist' % load_folder_path)

with open(os.path.join(load_folder_path, 'tags_to_array.pkl'), 'rb') as handle:
    tags_to_array = pickle.load(handle)
    slots_num = len(tags_to_array.label_encoder.classes_)

model = BertSlotModel.load(load_folder_path, sess)
tokenizer = FullTokenizer(vocab_file=vocab_file)

while True:
    print('\nEnter your sentence: ')
    try:
        input_text = input().strip()

    except:
        continue

    if input_text == 'quit':
        break

    input_text = ' '.join(tokenizer.tokenize(input_text))

    #벡터화
    data_text_arr = [input_text]
    print(data_text_arr)
    data_input_ids, data_input_mask, data_segment_ids = bert_to_array.transform(data_text_arr)

    #예측 결과 출력
    inferred_tags, slots_score = model.predict_slots([data_input_ids, data_input_mask, data_segment_ids], tags_to_array)
    print("Inferred tags")
    print(inferred_tags)
    print("Slots score")
    print(slots_score)        
    

    

tf.compat.v1.reset_default_graph()