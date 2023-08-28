from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from data import *
import torch



# token level idx --> 을 구하면 된다.
# 토큰 마다 --> word + token (쉬운버전) / char + token (쉬운버전)
    


class KlueDataset(Dataset):
    def __init__ (self, contexts, questions, answers, tokenizer):
        self.tokenizer = tokenizer
        self.answers = answers
        self.questions = questions
        self.contexts = contexts
        self.encodings = self.tokenizer(self.contexts, self.questions, max_length = 512, 
                      truncation = True, padding = 'max_length', return_token_type_ids = False)
        self.add_token_positions()
        
        
    def add_token_positions(self):
        start_positions = []
        end_positions = []
        
        
        for i in range(len(self.answers)):
        
            if 'answer_start' in self.answers[i]:
                if self.encodings.char_to_token(i, self.answers[i]['answer_start'][0]) == None :
                    start_positions.append(int(0))
                else:
                    start_positions.append(self.encodings.char_to_token(i, self.answers[i]['answer_start'][0]))
        
        
            if 'answer_end' in self.answers[i]:
                if self.encodings.char_to_token(i, self.answers[i]['answer_end'][0]) == None :
                    end_positions.append(int(0))
                else : 
                    end_positions.append(self.encodings.char_to_token(i, self.answers[i]['answer_end'][0]))
            else:
                end_positions.append(int(0))
        
        self.encodings.update({'start_positions' : start_positions, 'end_positions' : end_positions})
        
        
    
    def get_data(self):
        return None
    
    def __getitem__(self,idx): 
        
        
        return {'input_ids' : torch.tensor(self.encodings['input_ids'][idx], dtype = torch.long),
                'attention_mask' : torch.tensor(self.encodings['attention_mask'][idx], dtype = torch.long),
                'start_position' : torch.tensor(self.encodings.start_positions[idx], dtype = torch.long),
                'end_position' : torch.tensor(self.encodings.end_positions[idx], dtype = torch.long)
                }

    def __len__(self) :
        return len(self.encodings['input_ids'])
               
        

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base-uncased")
    train, validation = get_qa_data()
    train_contexts, train_questions, train_answers = preprocess(train)

    dataset = KlueDataset(train_contexts, train_questions, train_answers, tokenizer)
    print(dataset[0])




tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# a = tokenizer(train_contexts[0], train_questions[0])
# b = tokenizer.decode(a['input_ids'])
# print(b)
# print(train_contexts[0])

# 토큰화가 다 진행되었다면 다음으로 answer의 시작/종료 위치를 토큰화된 context 안에서 answer의 시작/종료 위치로 바꿔야 합니다.
# 이를 위해 char_to_token을 사용하였는데, 이는 원래의 문자열에서 토큰의 인덱스를 가져오는 역할을 합니다.


# encodings = tokenizer(train_contexts, train_questions, max_length = 512, 
#                       truncation = True, padding = 'max_length', return_token_type_ids = False)

# start_positions = []
# end_positions = []




# # 흠..이건 batch size 가 아니라 그냥 char_index 만 줘야할까?
# print(train_answers[1]['answer_start'][0])
# print(encodings.char_to_token(1, train_answers[1]['answer_start'][0]))
# train_answers = add_end_idx(train_answers, train_contexts)
# # print(train_answers)




# for i in range(len(train_answers)):
#     start_positions.append(encodings.char_to_token(train_answers[i]['answer_start'][0]))
    
#     if 'answer_end' in train_answers[i]:
#         end_positions.append(encodings.char_to_token(train_answers[i]['answer_end'][0]))
#     else:
#         end_positions.append(None)
    
# encodings.update({'start_positions' : start_positions, 'end_positions' : end_positions})

