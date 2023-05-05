#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[5]:


data = pd.read_csv('names.txt', header=None)
data.columns = ['name']
data


# In[6]:


with open('names.txt') as f:
    names = [name.strip() for name in f.readlines()]
    
names[:5]


# **Биграмма** букв - это пара букв, которые стоят рядом в слове. Например, в слове "кот" биграммы это "^к", "ко", "от", и "т$" (*^ и $ — начало и конец слова.*)
# 
# Мы используем биграммы букв, чтобы лучше понимать, какие пары букв чаще всего встречаются в словах. Это может быть полезно, например, для предсказания следующего слова в предложении или определения языка, на котором написан текст. 
# 
# Представьте что вам нужно придумать имя для вашего ребенка. У вас есть список уже существующих имен. Ваша задача: придумать **новое имя!**

# In[7]:


name = 'кот'
bag_of_bigrams = []
bag_of_bigrams.append(f'^{name[0]}')
for i in range(len(name)-1):
    bag_of_bigrams.append(f'{name[i]}{name[i+1]}')
    
bag_of_bigrams.append(f'{name[-1]}$')

bag_of_bigrams


# In[8]:


def bigram_transform(names):
    bag_of_bigrams = []
    for name in names:
        bag_of_bigrams.append(f'^{name[0]}')
        
        for i in range(len(name)-1):
            bag_of_bigrams.append(f'{name[i]}{name[i+1]}')
            
        bag_of_bigrams.append(f'{name[-1]}$')
        
    return bag_of_bigrams


bag_of_bigrams = bigram_transform(names)

bag_of_bigrams[:10]


# In[32]:


print('Total names:', len(names))
print('Total bigrams:', len(bag_of_bigrams))


# **Логика модели:**
# 
# - Прочитайте данные с файла в структуры данных удобных для высчитывания вероятностей
# - Высчитайте вероятность всех существующих биграмм (строим выборку)
# - Возьмите букву из выборки которое может придти как первая буква имени (рандомно)
# - Продолжать тянуть следующую букву из выборки, таким образом генерируя имя. Это нужно делать пока вы не вытянули конец имени.

# **Пользователь**
# 
# - Generate function - возможность создавать имя.
# - получить таблицу визуализирующие вероятности биграмм

# In[33]:


# there I've created part of probibility for generating names.

name_probability = {}

for bigram in bag_of_bigrams:
    first_letter, second_letter = bigram
    if first_letter not in name_probability:
        name_probability[first_letter] = {}
        
    if second_letter not in name_probability[first_letter]:
        name_probability[first_letter][second_letter] = 0
        
    name_probability[first_letter][second_letter] += 1
    
for first_letter in name_probability:
    sum_of_frequency = sum(name_probability[first_letter].values())
    for second_letter in name_probability[first_letter]:
        name_probability[first_letter][second_letter] /= sum_of_frequency
    
print(name_probability['^'])


# In[12]:


def next_letter_by_probability(letter='^', probability_dict={'^': {'*': 1.0}}):
    next_letter = np.random.choice(
        list(probability_dict[letter].keys()), 
        size=1, 
        replace=False, 
        p=list(probability_dict[letter].values())
    )[0]
    return next_letter


# In[15]:


def generate_function(probability_dict={'^': {'*': 1.0}}):
    first_letter = np.random.choice(list(name_probability.keys()))
    generated_name = f'{first_letter}'
    length_name = 0
    while first_letter != '$':
        old_letter = first_letter
        first_letter = next_letter_by_probability(first_letter, probability_dict)
        if first_letter == '$' and length_name < 3:
            first_letter = old_letter
            continue
        generated_name += first_letter
        length_name += 1
        
    return generated_name.replace('^', '').replace('$', '')


# In[29]:


print('30 generated names:')
for _ in range(30):
    generated_name = generate_function(name_probability)
    print('\t', generated_name)


# **Бонус**
# 
# - использование и оптимизирование с помощью библиотеки **pytorch**
# - визуализация таблицы в картинку (подойдет любая библиотека)

# ### __Rows mean the first letter, columns for the second part. e.g.__
# 
# ### __If we have some kind of letter "j", and after it there should be next letter with the highest probability is "a"(0.51%)__

# In[34]:


name_probability_df = pd.DataFrame(name_probability).sort_index().T.sort_index()
name_probability_df = name_probability_df.fillna(0)

sns.set(rc={'figure.figsize':(24, 16)})

sns.heatmap(name_probability_df.round(2), annot=True, cmap='Blues')
# name_probability_df.style.background_gradient(cmap='Blues')


# **2x Бонус**
# 
# Создать нейронную сеть которая учится на выборке. Гугл в помощь!

# In[37]:


import torch

import string


# In[129]:


all_letters = f'{string.ascii_lowercase}$'
n_letters = len(all_letters)


# In[130]:


def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# In[131]:


import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# In[132]:


import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    line = randomChoice(names)
    return line


# In[138]:


def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)


# In[139]:


def randomTrainingExample():
    line = randomTrainingPair()
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return input_line_tensor, target_line_tensor


# In[140]:


criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


# In[141]:


import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# In[142]:


rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


# In[143]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)
plt.show()


# In[144]:


def test(start_letter='A', max_length=20):
    with torch.no_grad():
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name
    
for _ in range(30):
    first_letter = np.random.choice(list(all_letters))
    
    generated_name = test(first_letter, max_length=20)
    print('\t', generated_name)
    


# In[ ]:




