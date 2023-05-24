## Randomly Retreive each set
import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import random
import numpy as np
import torchrimport os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification, AutoModel
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR

import json

# https://ddnews.co.kr/mbti-%EC%A7%88%EB%AC%B8-%EB%A6%AC%EC%8A%A4%ED%8A%B8/ 13, 16, 17, 14
q_category = [0, 1, 2, 3, 2, 0, 3, 2, 3, 2, 0, 1, 2, 3, 0, 0, 1, 2, 1, 3, 0, 1, 2, 3, 2, 0, 2, 1, 3, 1, 0, 1, 2, 3, 1, 0, 1, 2, 3, 1, 0, 2, 0, 3, 1, 1, 2, 2, 3, 1, 0, 1, 0, 2, 1, 3, 2, 2, 3, 3]

questions= """주기적으로 새로운 친구를 만든다.
자유 시간 중 상당 부분을 다양한 관심사를 탐구하는 데 할애한다.
다른 사람이 울고 있는 모습을 보면 자신도 울고 싶어질 때가 많다.
일이 잘못될 때를 대비해 여러 대비책을 세우는 편이다.
압박감이 심한 환경에서도 평정심을 유지하는 편이다.
파티나 행사에서 새로운 사람에게 먼저 자신을 소개하기보다는 주로 이미 알고 있는 사람과 대화하는 편이다.
하나의 프로젝트를 완전히 완료한 후 다른 프로젝트를 시작하는 편이다.
매우 감상적인 편이다.
일정이나 목록으로 계획을 세우는 일을 좋아한다.
작은 실수로도 자신의 능력이나 지식을 의심하곤 한다.
관심이 가는 사람에게 다가가서 대화를 시작하기가 어렵지 않다.
예술 작품의 다양한 해석에 대해 토론하는 일에는 크게 관심이 없다.
감정보다는 이성을 따르는 편이다.
하루 일정을 계획하기보다는 즉흥적으로 하고 싶은 일을 하는 것을 좋아한다.
다른 사람에게 자신이 어떤 사람으로 보일지 걱정하지 않는 편이다.
단체 활동에 참여하는 일을 즐긴다.
결말을 자신의 방식으로 해석할 수 있는 책과 영화를 좋아한다.
자신보다는 남의 성공을 돕는 일에서 더 큰 만족감을 느낀다.
관심사가 너무 많아 다음에 어떤 일을 해야 할지 모를 때가 있다.
일이 잘못될까봐 자주 걱정하는 편이다.
단체에서 지도자 역할을 맡는 일은 가능한 피하고 싶다.
자신에게 예술적 성향이 거의 없다고 생각한다.
사람들이 감정보다는 이성을 중시했다면 더 나은 세상이 되었으리라고 생각한다.
휴식을 취하기 전에 집안일을 먼저 끝내기를 원한다.
다른 사람의 논쟁을 바라보는 일이 즐겁다.
다른 사람의 주의를 끌지 않으려고 하는 편이다.
감정이 매우 빠르게 변하곤 한다.
자신만큼 효율적이지 못한 사람을 보면 짜증이 난다.
해야 할 일을 마지막까지 미룰 때가 많다.
사후세계에 대한 질문이 흥미롭다고 생각한다.
혼자보다는 다른 사람과 시간을 보내고 싶어한다.
이론 중심의 토론에는 관심이 없으며 원론적인 이야기는 지루하다고 생각한다.
자신과 배경이 완전히 다른 사람에게도 쉽게 공감할 수 있다.
결정을 내리는 일을 마지막까지 미루는 편이다.
이미 내린 결정에 대해서는 다시 생각하지 않는 편이다.
혼자만의 시간을 보내기보다는 즐거운 파티와 행사로 일주일의 피로를 푸는 편이다.
미술관에 가는 일을 좋아한다.
다른 사람의 감정을 이해하기 힘들 때가 많다.
매일 할 일을 계획하는 일을 좋아한다.
불안함을 느낄 때가 거의 없다.
전화 통화를 거는 일은 가능한 피하고 싶다.
자신의 의견과 매우 다른 의견을 이해하기 위해 많은 시간을 할애하곤 한다.
친구에게 먼저 만나자고 연락하는 편이다.
계획에 차질이 생기면 최대한 빨리 계획으로 돌아가기 위해 노력한다.
오래전의 실수를 후회할 때가 많다.
인간의 존재와 삶의 이유에 대해 깊이 생각하지 않는 편이다.
감정을 조절하기보다는 감정에 휘둘리는 편이다.
상대방의 잘못이라는 것이 명백할 때도 상대방의 체면을 살려주기 위해 노력한다.
계획에 따라 일관성 있게 업무를 진행하기보다는 즉흥적인 에너지로 업무를 몰아서 처리하는 편이다.
상대방이 자신을 높게 평가하면 나중에 상대방이 실망하게 될까 걱정하곤 한다.
대부분의 시간을 혼자서 일할 수 있는 직업을 원한다.
철학적인 질문에 대해 깊게 생각하는 일은 시간 낭비라고 생각한다.
조용하고 사적인 장소보다는 사람들로 붐비고 떠들썩한 장소를 좋아한다.
상대방의 감정을 바로 알아차릴 수 있다.
무엇인가에 압도당하는 기분을 느낄 때가 많다.
단계를 건너뛰는 일 없이 절차대로 일을 완수하는 편이다.
논란이 되거나 논쟁을 불러일으킬 수 있는 주제에 관심이 많다.
자신보다 다른 사람에게 더 필요한 기회라고 생각되면 기회를 포기할 수도 있다.
마감 기한을 지키기가 힘들다.
일이 원하는 대로 진행될 것이라는 자신감이 있다.""".split("\n")

# FIX RANDOM SEED
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def convert_MBTI(x):
    tmp = "ISTJ"
    return [0 if letter in tmp else 1 for letter in x]


class ClfModel(nn.Module):
    def __init__(self, name_or_path, out_size=1024):
        super(ClfModel, self).__init__()
        self.model = AutoModel.from_pretrained(name_or_path)
        self.out = out_size
        
        self.nn_1 = nn.Sequential(
         nn.Dropout(p=0.1),
         nn.Linear(self.out, 1)
        )
    
    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:, 0, :] # select first token.
        out1 = self.nn_1(out)

        return out1

def collate_fn(batch):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []

    # Pad all sequences to the same length
    max_len = max([x['input_ids'].size(1) for x in batch])

    for x in batch:
        input_ids.append(torch.nn.functional.pad(x['input_ids'], (0, max_len - x['input_ids'].size(1)), value=0))
        attention_mask.append(torch.nn.functional.pad(x['attention_mask'], (0, max_len - x['attention_mask'].size(1)), value=0))
        token_type_ids.append(torch.nn.functional.pad(x['token_type_ids'], (0, max_len - x['token_type_ids'].size(1)), value=0))
        labels.append(x['label'])

    # Stack padded sequences into tensors
    input_ids = torch.stack(input_ids).squeeze()
    attention_mask = torch.stack(attention_mask).squeeze()
    token_type_ids = torch.stack(token_type_ids).squeeze()
    labels = torch.stack(labels).squeeze()
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'labels': labels}


def get_acc(preds, labels, threshold=0.5, show_result=False):

    sig = nn.Sigmoid()

    with torch.no_grad():
        new_preds = sig(preds)  
    
    new_preds = new_preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    binary_preds = (new_preds >= threshold).astype(np.float32)

    # Compute accuracy using the binary predictions and true targets
    # correct_preds = (binary_preds == labels).all(axis=1).astype(np.float32)
    # strong_accuracy = correct_preds.sum() / len(labels)

    weak_preds = (binary_preds == labels).mean().astype(np.float32)
    # weak_accuracy = weak_preds.sum() / len(labels)

    if show_result:
        return weak_preds, binary_preds
    
    return weak_preds.item()




def run(THIS_IDX):

    class MBTIDataset(Dataset):
        def __init__(self, df, tokenizer, mode="Train"):
            self.df = df
            self.tokenizer = tokenizer
            self.users = self.df.User_ID.unique()

            self.mode = mode

        def __len__(self):
            return len(self.users)

        def __getitem__(self, idx):
                
            user_id = self.users[idx]
            user_data = self.df[self.df.User_ID == user_id]
            chosen = user_data.sample(n=N, replace=False, random_state=None)
            
            tmp = chosen['Final_Answer'].tolist()
            
            random.shuffle(tmp)
            text = '\n\n'.join(tmp)
            
            label = [convert_MBTI(chosen['MBTI'].unique()[0])[THIS_IDX]]
            encoding = self.tokenizer(text, max_length=384, padding='max_length', truncation=True, return_tensors='pt')

            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding['token_type_ids'],
                'label': torch.FloatTensor(label).unsqueeze(0)
            }


    ### MAIN CODE ### 
    main_df = pd.read_csv("/workspace/final_QIA/merged_train.csv")
    test_df = pd.read_csv("/workspace/final_QIA/phase2/test.csv")
    # questions = pd.read_excel("/workspace/COMP/QIA/Question.xlsx")

    main_df['Answer'] = main_df['Short_Answer'] + " " + main_df['Long_Answer']
    main_df['Q_text'] = main_df['Q_number'].map(lambda x: questions[x-1])
    main_df['Final_Answer'] = main_df['Long_Answer'] + " [SEP]"

    test_df['Answer'] = test_df['Short_Answer'] + " " + test_df['Long_Answer']
    test_df['Q_text'] = test_df['Q_number'].map(lambda x: questions[x-1])
    test_df['Final_Answer'] = test_df['Long_Answer'] + " [SEP]"
    

    rseed = 42
    set_seed(rseed)

    tgt = list(main_df['User_ID'].unique())

    train = []
    test = []


    q_list = [i + 1 for i in range(60) if q_category[i] == THIS_IDX]
    mbtis = [main_df[main_df.User_ID == x].iloc[0]['MBTI'] for x in tgt]
    mbti_labels = [0 if x[THIS_IDX] in "ISTJ" else 1 for x in mbtis] 
    

    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rseed)
    kf = [x for x in kf.split(tgt, mbti_labels)]

    train = [tgt[i] for i in kf[0][0]] # only use the first seed. -> second seed
    val =  [tgt[i] for i in kf[0][1]] # only use the first seed. -> second seed

    # # Set phase1 as train and phase2 as val
    train = list(range(1, 241))    
    val = list(range(241, 361))

    # from collections import Counter
    # a = Counter([mbti_labels[x] for x in kf[0][0]])
    # b = Counter([mbti_labels[x] for x in kf[0][1]])

    main_df = main_df[main_df.Q_number.isin(q_list)]
    test_df = test_df[test_df.Q_number.isin(q_list)]

    # Assign random MBTIs for test set.
    test_df['MBTI'] = test_df['Answer'].map(lambda x: "ISTJ")

    # # Separate Train and Valid
    train_df = main_df[main_df.User_ID.isin(train)]
    val_df = main_df[main_df.User_ID.isin(val)]

    N = 10

    model_name = "klue/roberta-large"
    model_path = f"/workspace/final_QIA/models_phase3/{model_name}"

    os.makedirs(model_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ClfModel(model_name, out_size=1024)

    train_dataset = MBTIDataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    val_dataset = MBTIDataset(val_df, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    test_dataset = MBTIDataset(test_df, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    sig = nn.Sigmoid()

    from tqdm import tqdm
    model.to(device)

    def are_tensors_equal(tensor_list):
        first_tensor = tensor_list[0]
        for tensor in tensor_list[1:]:
            if not torch.equal(first_tensor, tensor):
                return False
        return True

    # effective batch size = 32: 4 * 8
    best_acc = 0

    for epoch in tqdm(range(5)):
        
        # Basic initital settings.
        train_loss, test_loss, train_wacc, train_sacc, test_wacc, test_sacc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        ### TRAINING ###
        model.train()

        times = 0
        NUM_REP = 10
        
        # Repeat 5 times at least.
        for i in tqdm(range(NUM_REP)):
            for batch in train_dataloader:            
                optimizer.zero_grad()

                output = model(
                    input_ids= batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device)
                )
                
                loss = criterion(output.squeeze(), batch['labels'].to(device))
                loss.backward()

                train_loss += loss.item()

                train_wacc += get_acc(output.squeeze(), batch['labels'].to(device), 0.5)

                times += 1

                optimizer.step() 

        # # Trivial Error
        avg_train_loss = train_loss / times
        avg_train_wacc = train_wacc / times

        print('\n\n')
        print("----" * 30)
        print()
        print(f"Epoch {epoch}:")
        print("Train Loss: {:.4f}".format(avg_train_loss))
        print("Train Weak Acc: {:.4f}".format(avg_train_wacc))
        
        # ### VALIDATION ###
        model.eval()

        times = 0

        with torch.no_grad():
            
            soft_all_preds = []
            hard_all_preds = []

            for _ in range(3):
                
                all_results = []
                all_labels = []
                
                for _ in range(15):
                    
                    tmp_results = []
                    tmp_labels = []

                    for batch in val_dataloader:
                        with torch.no_grad():
                            output = model(
                                            input_ids= batch["input_ids"].to(device),
                                            attention_mask = batch["attention_mask"].to(device)
                                        )
                                        
                            
                            loss = criterion(output.squeeze(), batch['labels'].to(device))

                        test_loss += loss.item()
                        twacc = get_acc(output.squeeze(), batch['labels'].to(device), 0.5)

                        tmp_results.extend(output.squeeze().detach().cpu())
                        tmp_labels.extend(batch['labels'])

                        test_wacc += twacc
                        times += 1
                    
                    all_results.append(torch.stack(tmp_results))
                    all_labels.append(torch.stack(tmp_labels))
                
                if not are_tensors_equal(all_labels):
                    raise ValueError
                
                # Soft Voting
                final_results_soft = sum(all_results) / len(all_results)
                final_results_soft = [(sig(x) >= 0.5).detach().cpu().numpy().astype(np.float32).item() for x in final_results_soft]

                # Hard Voting 
                final_results_hard = [(sig(x) >= 0.5).detach().cpu().numpy().astype(np.float32) for x in all_results]
                final_results_hard = sum(final_results_hard) / len(final_results_hard)
                final_results_hard = [(x >= 0.5).astype(np.float32).item() for x in final_results_hard]

                soft_pred = sum((all_labels[0] == torch.Tensor(final_results_soft)).numpy().astype(np.float32)) / len(final_results_soft)
                hard_pred = sum((all_labels[0] == torch.Tensor(final_results_hard)).numpy().astype(np.float32)) / len(final_results_hard)

                soft_all_preds.append(soft_pred)
                hard_all_preds.append(hard_pred)

        avg_test_loss = test_loss / times
        avg_test_wacc = test_wacc / times
        

        print(f"Epoch {epoch}:")
        print("Test Loss: {:.4f}".format(avg_test_loss))
        print("Test Weak Acc: {:.4f}".format(avg_test_wacc))
        print("Tested", NUM_REP * 10, "samples")
        
        print("SOFT PRED:", sum(soft_all_preds) / len(soft_all_preds), "\n", soft_all_preds)
        print("HARD PRED:", sum(hard_all_preds) / len(hard_all_preds), "\n", hard_all_preds)

        ## Saving
        ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_acc': avg_train_wacc,
                'test_acc': sum(hard_all_preds) / len(hard_all_preds),
            }

        # SET BEST ACC AS HARD ACC
        if sum(hard_all_preds) / len(hard_all_preds) > best_acc:
            best_acc = sum(hard_all_preds) / len(hard_all_preds)
            torch.save(ckpt, f"{model_path}/{THIS_IDX}.pth")
            
            # TEST
            print("\nGenerating Test Set...\n")

            with torch.no_grad():
     
                all_results = []
                all_labels = []
                

                for _ in tqdm(range(100)):
                    
                    tmp_results = []

                    for batch in test_dataloader:
                        with torch.no_grad():
                            output = model(
                                            input_ids= batch["input_ids"].to(device),
                                            attention_mask = batch["attention_mask"].to(device)
                                        )
                                        
                            
                            loss = criterion(output.squeeze(), batch['labels'].to(device))

                        tmp_results.extend(output.squeeze().detach().cpu())
                    
                    all_results.append(torch.stack(tmp_results))
                    all_labels.append(torch.stack(tmp_labels))
    
                # Soft Voting
                final_results_soft = sum(all_results) / len(all_results)
                final_results_soft = [(sig(x) >= 0.5).detach().cpu().numpy().astype(np.float32).item() for x in final_results_soft]

                # Hard Voting 
                final_results_hard = [(sig(x) >= 0.5).detach().cpu().numpy().astype(np.float32) for x in all_results]
                final_results_hard = sum(final_results_hard) / len(final_results_hard)
                final_results_hard = [(x >= 0.5).astype(np.float32).item() for x in final_results_hard]

                with open(f"phase3_{THIS_IDX}.json", "w") as f:
                    f.write(json.dumps(final_results_hard)) 
                

if __name__ == "__main__":
    for i in range(4):
        run(i)