# -*- coding: utf-8 -*-
from data import *
from dataset import *
from model import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
import os


batch_size = 32
plm = 'deepset/bert-base-cased-squad2'
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")


# train, validation = get_qa_data()
# train = train [:250]
# test = validation[51:52]
# validation = validation [:50]

# train_contexts, train_questions, train_answers = preprocess(train)
# validation_contexts, validation_questions, validation_answers = preprocess(validation)


# train_dataset = KlueDataset(train_contexts, train_questions, train_answers, tokenizer)
# val_dataset = KlueDataset(validation_contexts, validation_questions, validation_answers, tokenizer)

# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


# model = QuestionAnswering(plm, 2)

# trainer = pl.Trainer(
#         max_epochs=1,
#         logger=TensorBoardLogger(save_dir='logs/', name='ner_model')
#     )

# trainer.fit(model, train_dataloader, val_dataloader)


# os.makedirs('saved_models', exist_ok=True)
# trainer.save_checkpoint('saved_models/QA_model.ckpt')


model = QuestionAnswering.load_from_checkpoint('/Users/margokim/Documents/pytorch/QA_Klue/saved_models/QA_model.ckpt', plm=plm, num_labels = 2)

model.eval()

input_text = '튀코 브라헤(Tycho Brahe, 1546년 12월 14일 ~ 1601년 10월 24일)는 덴마크의 천문학자이다. 로스토크 대학에서 공부했다. 1572년, 카시오페이아자리에서 신성을 발견하여 맨눈으로 관찰할 수 없을 때까지 14개월간 관측을 계속하여 기록을 남겼다. 이러한 재능이 인정되어 덴마크왕 덴마크의 프레데리크 2세 의 지원을 받아 벤 섬에 우라니보르 천문대, 스티에르네보르 천문대를 건설하고 방대하고 정밀한 관측기록을 남긴다. 프레데리크 2세가 죽은 후, 1599년에는 신성 로마 제국 황제 루돌프 2세 황실부 제국수학관에 초청되어 프라하로 이주한다. 천문학자로서 그는 코페르니쿠스 체계(지동설)의 기하학적 장점과 프톨레마이오스 체계(천동설)의 철학적 장점을 결합하여 태양이 지구 둘레를 도는 동시에 다른 행성들이 태양 둘레를 돈다는 독특한 튀코 체계(수정된 천동설)를 주장하였고, 이 학설의 선취권 문제로 라이마루스 우르소와 싸움을 벌였다. 그러나 브라헤가 천동설을 옹호하기 위해 남긴 관측기록은 그가 병으로 죽은 후 제자이며 공동연구자였던 요하네스 케플러가 그의 기록을 분석하여 케플러 법칙을 발견해 내면서 지동설을 지지하는 결정적인 증거가 되었다. 튀코 브라헤는 1577년 나타난 혜성에 대해서도 많은 관측결과를 남겼으며, 혜성 현상이 달보다 먼 곳에서 일어났다는 사실을 증명해 냈다. 그 혜성관측결과와 신성의 발견은 달보다 먼 곳에서는 어떠한 변화도 일어나지 않고 있다는 당시의 천동설을 반증하는 한 증거이기도 했다.'
input_question = '튀코 브라헤는 프레데리크 2세가 사망한 후 어디로 이주하나요?'
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
encodings = tokenizer(input_text, input_question , max_length = 512, 
                      truncation = True, padding = 'max_length', return_token_type_ids = False)


input_ids = torch.tensor(encodings['input_ids']).unsqueeze(0)
attention_mask = torch.tensor(encodings['attention_mask']).unsqueeze(0)

# with torch.no_grad():
#     start_idx, end_idx = model(input_ids, attention_mask)
    
# start_pos = torch.argmax(start_idx)
# end_pos = torch.argmax(end_idx)



with torch.no_grad():
    start_scores, end_scores = model(input_ids, attention_mask)
start_pos = torch.argmax(start_scores)
end_pos = torch.argmax(end_scores)
print(start_pos)
print(end_pos)


answer = tokenizer.decode(input_ids[0][start_pos])

print(answer)

