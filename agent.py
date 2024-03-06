from torch.distributions import Categorical
import random
import numpy as np
from transformers import AdamW, BertModel, RobertaModel, AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from utils import *
from prompt import ESConvAct, CIMAAct, CBAct

model = {'bert': BertModel, 'roberta': RobertaModel}
act = {'esc': ESConvAct, 'cima': CIMAAct, 'cb': CBAct}
TMP_DIR = {
    'esc': './tmp/esc',
    'cima': './tmp/cima',
    'cb': './tmp/cb',
}

class PPDPP(nn.Module):
    def __init__(self, args, config, tokenizer):
        super().__init__()
        self.policy = model[args.model_name].from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)
        self.dropout = nn.Dropout(0.5)
        self.act = sorted(list(act[args.data_name].keys()))
        self.classifier = nn.Linear(config.hidden_size, len(self.act))
        self.tokenizer = tokenizer
        self.optimizer = AdamW(
            self.parameters(), lr=args.learning_rate
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.config = config
        self.args = args
        self.saved_log_probs = []
        self.rewards = []

    def build_input(self, state):
        dial_id = []
        for turn in state[::-1]:
            s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
            if len(dial_id) + len(s) > self.args.max_seq_length:
                break
            dial_id = s[1:] + dial_id
        inp = s[:1] + dial_id
        return [inp]

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, len(self.act)), labels.view(-1))
            return loss
        else:
            return F.softmax(logits, dim=-1)

    def select_action(self, state, is_test=False):
        inp = self.build_input(state)
        inp = torch.tensor(inp).long()

        outputs = self.policy(inp)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = nn.functional.softmax(logits, dim=1)
        m = Categorical(probs)
        if is_test:
            action = probs.argmax().item()
        else:
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
        return self.act[action]

    def optimize_model(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if rewards.shape[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        return policy_loss.data
    
    def save_model(self, data_name, filename, epoch_user):
        output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
    def load_model(self, data_name, filename, epoch_user=None):
        if epoch_user: 
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        else:
            output_dir = filename
        if hasattr(self, 'module'):
            self.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            self.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location='cuda:0'))

