# Turkish-CRF-NER

Conditional Random Fields is a class of discriminative models best suited to prediction tasks where contextual information or state of the neighbors affect the current prediction. CRFs find their applications in named entity recognition, part of speech tagging, gene prediction, noise reduction and object detection problems, to name a few.

## Dataset

The datasets which I used you can find [here](https://drive.google.com/file/d/1-CYBDhE6dnM1kdnwF3Ue6ZMKeSyOGuZz/view?usp=sharing) this data is actually reduced (%50) frome the original dataset and added "sentence_Number" column. To train it I had to reduce it again and now the dataset is %47 of the original dataset. There are 4 different entities and 25 different labels we have. 


## Usage
First we need to import libraries

```python
import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```
After that we must implement our CSV file but there will bu Null values so we must use dropna() for that.

```python
df=pd.read_csv('/content/drive/My Drive/SoftTech/CRF/Dataset/HALF_FULL1_title_Test.csv')

df=df.dropna()
df.isnull().sum()
```
Now we can control our unique values in our columns

```python
df['sentence_Number'].nunique(), df.words.nunique(), df.entity.nunique(), df.label.nunique()
```
```python
df.groupby('entity').size().reset_index(name='counts')
```
```python
df.groupby('label').size().reset_index(name='counts')
```
Now we need to split our data to sentence_Number and words

```python
X = df.drop('entity', axis=1)
X = X.drop('label', axis=1)
X.head() 
```
##CRF
Now we will use CRF but we need to take sentence and it's values one by one
```python
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
```
```python
class SentenceGetter(object):
    
  def __init__(self, data):
    self.n_sent = 1
    self.data = data
    self.empty = False
    agg_func = lambda s: [(l, e, w) for l, e, w in zip( 
                                                        
                                                        s['label'].values.tolist(),
                                                        s['words'].values.tolist(),
                                                        s['entity'].values.tolist())]
    self.grouped = self.data.groupby('sentence_Number').apply(agg_func)
    self.sentences = [s for s in self.grouped]
      
  def get_next(self):
    try: 
      s = self.grouped[self.n_sent]
      self.n_sent += 1
      return s 
    except:

      return None
```
```python
getter = SentenceGetter(df)
```
```python
sent = getter.get_next()
print(sent)
```
```python
sentences = getter.sentences
```
## Feature Extractions
now we must use Word2Feature to split a sentence to it's labels, entities and words.

```python
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
        
    else:
        features['BOS'] = True
    
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]
```

```python
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]
```
Now it's time to split train and test datas

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
```
```python
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
```
```python
y = df.entity.values
```
```python
classes = np.unique(y)
```
```python
classes = classes.tolist()
classes
```
```python
new_classes = classes.copy()
new_classes.pop()
new_classes
```
```python
y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred, average='weighted', labels = new_classes)
```
```python

print(metrics.flat_classification_report(y_test, y_pred, labels = new_classes))
```
with the last part you can see your scores. with my data you f1-core is %63 but this is the %47 part of the original data.
