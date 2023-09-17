import numpy as np
import os, json, torch, nltk, re
from collections import defaultdict
from typing import List, Tuple, Dict

from model import Model

nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()

nltk.download("stopwords")
stopWords = nltk.corpus.stopwords.words('english')


glob_device = ''
def build_model(device: str) -> Model:
    global glob_device
    glob_device = device
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel()
    # return RandomBaseline()


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


class StudentModel(Model):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self):
        self.read_files()
        self.myModel = SimpleModel(
                                vocab_length=len(self.word2index),
                                vocab=self.word2index
                                ).to(glob_device)
        self.load_weights()
        
            
    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!
        # try:
        test_dataset = self.process_data(sentence_pairs)
        self.myModel.eval()
        with torch.no_grad():
            results = []
            for data in test_dataset:
                inputs = {
                        'first line':   data['first line encoded'],
                        'first count':  data['first line counted'],
                        'first key':    data['first key index'],
                        'second line':  data['second line encoded'],
                        'second count': data['second line counted'],
                        'second key':   data['second key index'],
                        }
                
                logits = self.myModel(inputs)

                preds = np.argmax(logits, axis=1)
                bool_preds = np.array(preds > 0)
                for result in bool_preds:
                    results.append(str(result))
                
            return results
            
    def process_data(self, sentences):
        test = PrepareDataset(dict_input=sentences, 
                            vocab=[self.word2index, self.index2word],
                            max_length=self.max_length,
                            device=glob_device)

        return torch.utils.data.DataLoader(test, batch_size=16)

    def load_weights(self):
        path2best = r'model/0.6994.pt'
        checkpoint = torch.load(path2best, map_location=glob_device)
        self.myModel.load_state_dict(checkpoint['model_state_dict'])

    def read_files(self):

        two_up = os.path.dirname(os.path.dirname(os.getcwd()))
        with open(r'model/word2index.json', 'r') as f:
            self.word2index = json.load(f)

        with open(r'model/index2word.json', 'r') as f:
            self.index2word = json.load(f)

        with open(r'model/max_length.json', 'r') as f:
            self.max_length = json.load(f)

class SimpleModel(torch.nn.Module):
    def __init__(self, 
                 vocab_length=None,
                 vocab=None,
                 embedding=None,
                 dropout=0.5,
                 embedding_dim=50, 
                 hidden_dim=50,
                 device='cpu',
                 aggregate='separate'
                 ):
        
        super(SimpleModel, self).__init__()

        self.device = device
        self.aggregate = aggregate
        self.embedding_dim = embedding_dim

        self.word_embedding = torch.nn.Embedding(vocab_length, self.embedding_dim, padding_idx=vocab['<PAD>'])
        if embedding:
          self.word_embedding.weight.data.copy_(torch.stack(embedding))

        if 'separate' in self.aggregate:
          embedding_dim *= 2

        self.linear = torch.nn.Linear(embedding_dim, hidden_dim)
        
        # self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/2))

        self.dropout = torch.nn.Dropout(dropout)
        self.act_fn = torch.relu
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        

    def forward(self, x):
        o = self.aggregation_layer(x)
        o = self.linear(o)
        o = self.dropout(o)
        o = self.act_fn(o)

        return self.classifier(o)

    def aggregation_layer(self, x):

        first_x = x['first line']
        first_c = x['first count'].unsqueeze(1)
        first_k = x['first key']
        second_x = x['second line']
        second_c = x['second count'].unsqueeze(1)
        second_k = x['second key']

        first_output = self.word_embedding(first_x).to(self.device)
        second_output = self.word_embedding(second_x).to(self.device)

        if 'weighted' in self.aggregate:
          b = torch.arange(first_output.shape[0])
          f = first_k
          s = second_k

          first_output[b, f] *= 2
          second_output[b, s] *= 2


        first_sum = torch.sum(first_output, axis=1)
        first_mean = first_sum / first_c

        second_sum = torch.sum(second_output, axis=1)
        second_mean = second_sum / second_c

        if 'separate' in self.aggregate:
          return torch.cat([first_mean, second_mean], axis=1)
        else:
          return first_mean - second_mean

class PrepareDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 dict_input=None, 
                 vocab=None, 
                 model_type='simple', 
                 keyword_token_type='Lemma', 
                 number_token_type='None',
                 lemmatize=True, 
                 remove_stopwords=True, 
                 delete_freq=1, 
                 max_length=0, 
                 device='cpu'):
        
        self.dict_input = dict_input
        self.vocab = vocab
        self.model_type = model_type
        self.keyword_token = keyword_token_type
        self.number_token_type = number_token_type
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.delete_freq = delete_freq
        self.max_length = max_length
        self.device = device

        self.replacement_word = 'keywordreplacement'
        self.dataset = []
        self.word2index, self.index2word = {}, {}

        self.process_data()
    
    def __len__(self):  
        return len(self.processed)
    
    def __iter__(self):
        
        for data in self.processed:
            first_key = data['first key']
            first_line = data['first line']
            second_key = data['second key']
            second_line = data['second line']

            first_line_new = first_line.copy()
            first_key_index = first_line_new.index(self.replacement_word)
            first_line_new[first_key_index] = first_key

            second_line_new = second_line.copy()
            second_key_index = second_line_new.index(self.replacement_word)
            second_line_new[second_key_index] = second_key

            first_line_encoded = self.index_words(first_line_new)
            second_line_encoded = self.index_words(second_line_new)
            
            x = {
                'first key index': torch.tensor(first_key_index).to(self.device),
                # 'first key encoded': torch.tensor(self.word2index[first_key]).to(self.device),
                'first line encoded': torch.LongTensor(first_line_encoded).to(self.device),
                'first line counted': torch.tensor(len(first_line_new)).to(self.device),     
                'second key index': torch.tensor(second_key_index).to(self.device),
                # 'second key encoded': torch.tensor(self.word2index[second_key]).to(self.device),
                'second line encoded': torch.LongTensor(second_line_encoded).to(self.device),
                'second line counted': torch.tensor(len(second_line_new)).to(self.device),
            }

            yield x

    def index_words_lstm(self, line1, line2):
        recreate_line = []
        if self.delete_freq > 0 and self.vocab == None: # means it's working on training data
            line1 = self.delete_lower_freq(line1)
            line2 = self.delete_lower_freq(line2)

        lines = np.concatenate([line1, ['<EOS>'], line2])

        encoded = []
        for word in lines:
            encoded.append(self.word2index[word])

        line_len = len(encoded)

        while len(encoded) < self.max_length:
            encoded.append(self.word2index['<PAD>'])

        return encoded, line_len


    def index_words(self, line):
        recreate_line = []
        
        if self.delete_freq > 0 and self.vocab == None: # means it's working on training data
            line = self.delete_lower_freq(line)
      
        encoded = []
        for word in line:
            encoded.append(self.word2index[word])
            # encoded = [self.word2index[word] for word in line]

        while len(encoded) < self.max_length:
            encoded.append(self.word2index['<PAD>'])

        return encoded

    def delete_lower_freq(self, line):
        recreate_line = []
        for word in line:
            if self.freqs[word] >= self.delete_freq:
                recreate_line.append(word)
        return recreate_line

    def process_data(self):
        self.freqs = {}
        self.processed = []
        # lines = [[key_word1, splitted1], [key_word1, splitted1]]
        for lines in self.read_data():
            first_key, first_line = lines[0]['key'] ,lines[0]['line']
            second_key, second_line = lines[1]['key'], lines[1]['line']
            pos = lines[0]['pos']
            lemma = lines[0]['lemma']

            # tokenize keywords
            if self.keyword_token:
                first_key, second_key = self.keyword_tokenizer(first_key, second_key, pos, lemma)

            # remove_stopwords
            if self.remove_stopwords:
                first_line, second_line = self.stopwords_remover(first_line, second_line, [first_key, second_key])

            # tokenize sentences
            if self.lemmatize:
                first_line, second_line = self.lemmatizer(first_line, second_line)

            # discard numbers
            if self.number_token_type != None:
                first_line, second_line = self.number_tokenizer(first_line, second_line)

            # build vocabulary
            if self.vocab == None: # means it's working on training data
                self.build_vocab(lines)
            else:
                self.word2index, self.index2word = self.vocab[0], self.vocab[1]

            # delete freq
            if self.delete_freq > 0 and self.vocab == None:
                self.get_frequency([first_key, second_key], first_line, second_line)

            # max length
            if self.max_length < max(len(first_line), len(second_line)):
                self.max_length = max(len(first_line), len(second_line))
                
        
                    
            data = {
                'first key': first_key,
                'first line': first_line,
                'second key': second_key,
                'second line': second_line,
            }
            
            self.processed.append(data)
                
        self.word2index = defaultdict(lambda: 1, self.word2index)
        self.index2word = defaultdict(lambda: '<UNK>', self.index2word)

    def number_tokenizer(self, line1, line2):
        lines = []
        for line in [line1, line2]:
            pop_elem = []
            for i, word in enumerate(line):
                if word.isnumeric():
                    if self.number_token_type.lower() == "flag":
                        line[i] = "<NUMBER>"
                    elif self.number_token_type.lower() == "discard":
                        pop_elem.append(i)
        
            if len(pop_elem) > 0:
                for index in pop_elem[::-1]:
                    line.pop(index)
            lines.append(line)
        
        return lines[0], lines[1]

    def get_frequency(self, keys, line1, line2):
        lines = []
        for line in [keys, line1, line2]:
            for word in line:
                try:
                    self.freqs[word] += 1
                except:
                    self.freqs[word] = 1

    def stopwords_remover(self, line1, line2, keys):
        lines = []
        for index, line in enumerate([line1, line2]):
            if keys[index] not in stopWords:
                for word in line:
                    if word != self.replacement_word:
                        if word in stopWords:
                            i = line.index(word)
                            line.pop(i)

            lines.append(line)
      
        return lines[0], lines[1]
  
    def lemmatizer(self, line1, line2):	
        lines = []	
        for line in [line1, line2]:	
            lemmatized_words = []	
            for word in line:
                if word != self.replacement_word:
                    try:	
                        lemmatized_words.append(lemmatizer.lemmatize(word))	
                    except:	
                        lemmatized_words.append(word)	
                        print("Couldn't lemmatize: ", word)	
                else:
                    lemmatized_words.append(self.replacement_word)
                    
            lines.append(lemmatized_words)	

        return lines[0], lines[1]

    def keyword_tokenizer(self, key1, key2, pos, lemma):
        # if flagged, replace keyword with "TARGET"
        if self.keyword_token.lower() == "flag":
            key1, key2 = "<TARGET>", "<TARGET>"
        elif self.keyword_token.lower() == "pos":
            key1, key2 = pos, pos
        elif self.keyword_token.lower() == "lemma":
            key1, key2 = lemma, lemma

        return key1, key2


    def build_vocab(self, lines):
        try:
            assert self.word2index["<PAD>"]
        except:
            # print("aaaaa", self.word2index['<PAD>'])
            self.word2index["<PAD>"] = 0
            self.word2index["<UNK>"] = 1
            self.word2index["<EOS>"] = 2
            self.word2index["<TARGET>"] = 3
            self.index2word[0] = "<PAD>"
            self.index2word[1] = "<UNK>"
            self.index2word[2] = "<EOS>"
            self.index2word[3] = "<TARGET>"

        for line_dict in lines:
            key = line_dict['key']
            line = line_dict['line']
            all_words = np.append(line, key)
            for word in all_words:
                try:
                    assert word in self.word2index.keys()
                except:
                    self.word2index[word] = len(self.word2index)
                    self.index2word[len(self.index2word)] = word


    def read_data(self):    
        # df = pd.read_json(self.path, lines=True)
        df = self.dict_input

        sentence_ids = {
            "sentence1" : ["start1", "end1"],
            "sentence2" : ["start2", "end2"]
        }

        count_stopWords = 0
        count_all = 0
        # loop all rows in the dataframe
        
        for no, df_data in enumerate(df):

            lines = []
            pos = df_data['pos']
            lemma = df_data['lemma']

            # process each sentences in the same row
            for sentence_id in sentence_ids.keys():

                # get start and end indices of the related key word        
                start_id = sentence_ids[sentence_id][0]
                end_id = sentence_ids[sentence_id][1]
                
                # start_id => column, no => row    
                start = int(df_data[start_id])
                end = int(df_data[end_id])

                # find key word in the line and lower the letters
                key_word = df_data[sentence_id][start:end].lower()

                # get the corresponding line
                line = df_data[sentence_id]

                # count spaces to find exact position of the keyword
                # do this to know the pos of keyword
                # even if we delete some other words
                count = 1
                char_count = 0
                for word in line:
                    for letter in word:
                        char_count += 1
                        if letter == " " and start > char_count:
                            count += 1

                # replace keyword with a string        
                line = df_data[sentence_id][:start] + self.replacement_word + df_data[sentence_id][end:]
                
                splitted = [word for word in re.split('\W', line.lower()) if word]
                
            
                lines.append({'pos': pos, 'lemma': lemma, 'key': key_word, 'line': splitted})

            yield lines   
