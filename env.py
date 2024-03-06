import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template

import openai

from utils import *
from prompt import *
#from unidecode import unidecode
import nltk
import re
import time

system_role = {'esc':'Therapist', 'cima': 'Teacher', 'cb': 'Buyer'}
user_role = {'esc':'Patient', 'cima': 'Student', 'cb': 'Seller'}
message_format = {'esc': ESConvMessages, 'cima': CIMAMessages, 'cb': CBMessages}

YOUR_API_KEY = ""

class Env(object):
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
            if mode == 'train':
                self.vicuna_model, self.vicuna_tokenizer = load_model(
                    args.model_path,
                    args.device,
                    args.num_gpus,
                    args.max_gpu_memory,
                    args.load_8bit,
                    args.cpu_offloading,
                    debug=args.debug,
                )
            else:
                self.vicuna_model = env_model
                self.vicuna_tokenizer = env_tokenizer
        
        
        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.cur_conver_step = 0
        self.test_num = 0
        self.mode = mode

        self.reward_dict = {
            'esc': {
                'worse': -1.0,
                'same': -0.5,
                'better': 0.5,
                'solved': 1.0,
            },
            'cima': {
                'incorrect': -1.0,
                'did not': -0.5,
                'part': 0.5,
                'whole': 1.0,
            },
        }

        set_random_seed(args.seed)

        
    def reset(self):
        self.cur_conver_step = 0
        if self.mode == 'train':
            self.case = np.random.choice(self.dataset)
        elif self.mode == 'test':
            self.case = self.dataset[self.test_num]
            self.test_num += 1
        
        if self.args.data_name == 'esc':
            self.conversation = [{"role":"Patient", "content":self.case['situation']}]
        elif self.args.data_name == 'cima':
            self.conversation = [{"role":"Teacher", "content":self.case['dialog'][0]['text']}, {"role":"Student", "content":self.case['dialog'][1]['text']}]
        elif self.args.data_name == 'cb':
            self.conversation = [{"role":"Buyer", "content":"Hi, how much is the %s?" % self.case['item_name']}, {"role":"Seller", "content":"Hi, this is a good %s and its price is %s." % (self.case['item_name'], self.case['seller_price'])}]
        print(self.conversation)
        return self.conversation


    def step(self, action):
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        
        print(action)
        messages = message_format[self.args.data_name](self.case, 'system', self.conversation, action)
        response = self.generate_response(self.args.system, messages, system_role[self.args.data_name])
        response = self.postprocess_response(response, user_role[self.args.data_name])
        self.conversation.append({"role":system_role[self.args.data_name],"content":response})
        print(self.conversation[-1])

        messages = message_format[self.args.data_name](self.case, 'user', self.conversation)
        user_response = self.generate_response(self.args.user, messages, user_role[self.args.data_name])
        user_response = self.postprocess_response(user_response, system_role[self.args.data_name])
        self.conversation.append({"role":user_role[self.args.data_name], "content":user_response})
        print(self.conversation[-1])

        messages = message_format[self.args.data_name](self.case, 'critic', self.conversation)
        reward = self.compute_reward(self.args.critic, messages, self.case)

        if self.args.data_name == 'esc':
            if reward > 0.5:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
        elif self.args.data_name == 'cima':
            if reward == 1:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
        elif self.args.data_name == 'cb':
            if reward >= 0:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
                
        self.cur_conver_step += 1
        return self.conversation, reward, done
    
    def postprocess_response(self, response, role):
        #print(response)
        if role in response:
            response = response.split(role)[0].strip()
        sents = nltk.sent_tokenize(response)
        if len(sents) == 1:
            if response[-1] not in ['.','!','?',':']:
                return response + '.'
            return response.strip()
        try:
            if sents[-1].strip()[-1] not in ['.','!','?',':']:
                return ' '.join(sents[:-1]).strip()
            else:
                return response.strip()
        except Exception as e:
            return response.strip()

    def generate_response(self, model, messages, role):
        if self.mode == 'test':
            temperature = 0
        else:
            temperature = 0.7
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, role)
            #print(messages)
            output = query_openai_model(
                api_key=YOUR_API_KEY,
                messages=messages,
                model="gpt-3.5-turbo-0613",
                max_tokens=self.args.max_new_tokens,
                temperature=temperature
            )
        return output
    
    def compute_reward(self, model, messages, case):
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, user_role[self.args.data_name])
            outputs = query_openai_model(
                api_key=YOUR_API_KEY, 
                messages=messages,
                model="gpt-3.5-turbo-0613",
                max_tokens=self.args.max_new_tokens,
                temperature=1.1,
                n=10
            )
        
        if self.args.data_name in ['esc','cima']:
            rewards = []
            print(outputs)
            for output in outputs:
                for key in self.reward_dict[self.args.data_name]:
                    if key in output.lower():
                        rewards.append(self.reward_dict[self.args.data_name][key])
                        break
            if len(rewards) == 0:
                reward = 0
            else:
                reward = sum(rewards)/len(rewards)
            print(reward)
        elif self.args.data_name == 'cb':
            deals = []
            rewards = []
            print(outputs)
            for output in outputs:
                if 'have not' in output.lower():
                    deals.append(-1)
                elif 'have reached' in output.lower():
                    deals.append(1)
                
                prices = re.findall(r"[-+]?\d*\.?\d+", output.replace(",",""))
                if len(prices) > 0:
                    deal_price = float(prices[0])
                    reward = (deal_price - case['seller_price']) / (case['buyer_price'] - case['seller_price'])
                    rewards.append(reward)

            if -1 in deals:
                reward = -0.1
            else:
                if len(rewards) == 0:
                    reward = 0
                else:
                    reward = max(set(rewards), key = rewards.count)
            print(reward)

        return reward



def query_openai_model(api_key: str, messages: str, model: str = "gpt-3.5-turbo-0613", max_tokens: int = 128, temperature: float = 0, n: int = 1):
    openai.api_key = api_key
    flag = True
    while flag:
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=n,
                stop=None,
                temperature=temperature,
                request_timeout=10,
            )

            if n == 1:
                output = completions.choices[0].message.content.strip()
            else:
                output = []
                for choice in completions.choices:
                    output.append(choice.message.content.strip())

            flag = False
        except Exception as e:
            print("Some error happened here.")
            time.sleep(5)
    return output