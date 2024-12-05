## About TF-IDF
TF-IDF is a statistical measure used in information retrieval and text mining to evaluate the importance of a word in a document relative to a collection of documents (a corpus). It helps identify significant words while reducing the weight of commonly occurring ones.

### Key Components of TF-IDF
1. Term Frequency (TF):
   
    Measures how often a term appears in a document.
    Gives higher weight to frequently occurring terms in a specific document.

2. Inverse Document Frequency (IDF):
   
     Reduces the weight of common terms that appear in many documents (e.g., "the," "and").
     Ensures rare terms across the corpus get higher weight.

3. TF-IDF Score:
   
     Combines TF and IDF to compute a score for each term in a document.
     High TF-IDF scores indicate terms that are important to a particular document but uncommon in the entire corpus.


## Example
### Corpus
* "lion is eating meat"
* "the lion is running"
* "the dog is eating meat"

### Vocabulary
```python
["lion", "is", "eating", "meat", "the", "running", "dog"]
```
Vocabulary size is the length of the vector.

### TF Calculation
For the sentence "lion is eating meat", we calculate the term frequency for each word:

* "lion" appears once in the sentence, so TF = 1.
* "is" appears once in the sentence, so TF = 1.
* "eating" appears once in the sentence, so TF = 1.
* "meat" appears once in the sentence, so TF = 1.
* "the" does not appear in this sentence, so TF = 0.
* "running" does not appear in this sentence, so TF = 0.
* "dog" does not appear in this sentence, so TF = 0.

### IDF Calculation
* "lion" appears in 2 documents, so IDF = Log(3/2)
* "is" appears in 3 documents, so IDF = Log(3/3)
* "eating" appears in 2 documents, so IDF = Log(3/2)
* "meat" appears in 2 documents, so IDF = Log(3/2)
* "the" appears in 2 documents, so IDF = Log(3/2)
* "running" appears in 1 document, so IDF = Log(3/1)
* "dog" appears in 1 document, so IDF = Log(3/1)

```scss
log(3/2) ≈ 0.1761
log(3/3) = 0
log(3/2) ≈ 0.1761
log(3/2) ≈ 0.1761
log(3/2) ≈ 0.1761
log(3/1) ≈ 1.0986
log(3/1) ≈ 1.0986
```

### TF-IDF Vector for "Lion is eating meat"
```python
[0.1761, 0, 0.1761, 0.1761, 0, 0, 0]
```

## MLP

Model structure:
(input size must be same as the vector size)
```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)
```

* Input Size: Consistent with TF-IDF vectorizer's vocabulary size
* Hidden Size: Number of neurons in the middle layer
* Output Size: 2, for the binary classification

### Mini-Batch Training
* Definition:

  * The data is divided into small batches (e.g., 16, 32, or 64 samples per batch).
  * The model processes one batch at a time during training and updates the weights based on the average gradient of that batch.
  * Advantages:
    * Balanced Efficiency: Combines the benefits of stochastic and batch training.
    * Stable Convergence: Reduces noise in weight updates compared to stochastic training.
    * Parallelization: Leverages modern hardware like GPUs, which are optimized for batch processing.


