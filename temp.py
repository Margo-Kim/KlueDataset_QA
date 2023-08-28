from transformers import BertForQuestionAnswering


model = BertForQuestionAnswering.from_pretrained('klue/bert-base')

print()