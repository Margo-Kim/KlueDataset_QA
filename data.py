from datasets import load_dataset

def get_qa_data():
    dataset = load_dataset('klue', 'mrc')
    
    return dataset['train'], dataset['validation']


def preprocess(data):
    
    contexts = data['context']
    questions = data['question']
    answers = data['answers']
    
    return contexts, questions, answers


def add_end_idx(answers, contexts): 

    for i in range(len(answers)):
        gold_text = answers[i]['text']
        start_idx = answers[i]['answer_start']
        context = contexts[i]
        end_indices = [ ]
        is_error = False
        
        for text, idx in zip(gold_text, start_idx):
            end_idx = idx + len(text)
            
            if context[idx :end_idx] == text:
                end_indices.append(end_idx)
            else:
                # print(f"{context[idx :end_idx]} != {text}")
                is_error = True
                end_indices.append(0)
                break
            
        if is_error:
            continue
                
        answers[i]['answer_end'] = end_indices
        # append data
        
            
    return answers
            # if context[start_idx : end_idx] == text:
            #     answer[i]['answer_end'] = end_idx
            
            
            

train, validation = get_qa_data()

train_contexts, train_questions, train_answers = preprocess(train)

# print(add_end_idx(train_answers, train_contexts))
