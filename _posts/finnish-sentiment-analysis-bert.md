---
title: "Finnish Sentiment Analysis using BERT & Transformers."


excerpt: "Natural Language Processing, Sentiment Classification"
mathjax: "false"
---

# Introduction 

In this article we will develop a sentiment classifier on Finnish text using FinBERT by TurkuNLP and Transformers by Haggingface, and serve it as a web app using Streamlit or as an API using Uvicorn.


# What is BERT ?

BERT stands for Bidirectional Encoder Representations from Transformers, it was [published](https://arxiv.org/pdf/1810.04805.pdf) by researchers at Google and considered as the state of the art of NLP in wide variaty of tasks. it was built on the top of many innovative ideas introduced in the NLP community such as [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432), [ULMFiT](https://arxiv.org/abs/1801.06146), and Transformers ([OpenAI's](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) and [Vaswani et al's](https://arxiv.org/pdf/1706.03762.pdf)). 

Overall, BERT is a trained Transformer Encoder stack, Google's paper presented two model sizes, BERT-base: 12 Encoders with 12 bidirectional self-attention heads and BERT-large: 24 Encoders with 24 bidirectional self-attention heads. For a detaild BERT illustration and explaination, check [this amazing article by Jay Almmar](http://jalammar.github.io/illustrated-bert/).

Briefly, Transfomer is an attention mechanism that learns contextual relationship between words. Its encoder reads a sequence of words entirely at once through a self-attention layer that helps the encorder to look at other words in the input sequence as it encodes a specific word. The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position. The decoder has also a self-attention layer and feed forward layer , but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence (similar what attention does in seq2seq models). to learn more about Transformers, check this incredible [illustrated explaination](https://jalammar.github.io/illustrated-transformer/).

# Data Exploration & Preparation

Let's take a look at our dataset, I scraped some reviews data from Google Play by focusing on Finnish apps/language only.I tried to keep the dataset balanced as possible.

you can download the prepared dataset directly from my Drive. Make sure you have gdown installed and run the following command:

```
!gdown --id 1yGYZcIsFgTRd3aiMjIqWgH0KiHFxorAB
```
 the dataset contains 82450 finnish review and distributed as follows :

 <img src="/img/blog/senti-fi/sentiment_dist.PNG" alt="Sentiment dataset">

 You will find two columns representing the content of each review, one is raw, straigh from the Google Play (`content`), and an other one cleaned (`content_cleaned`) using some essential preprocessing steps summerized in the following function :

```python
import re

def clean_text(text):
    # Remove all the special characters
    text = re.sub(r'\W', ' ', str(text))

    # Remove URLS
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) 

    # remove all single characters
    text= re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 

    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)

    # Converting to Lowercase
    text = text.lower()
    
    ##Stemming & remove Stop words # YOU CAN UNCOMMENT THIS IS YOU WANT TO TEST WITH STEMMING.
    #text = [STEMMER.stem(word) for word in text] 

    return text
```

Before feeding our text dataset to BERT. each review should be split into tokens and each token should be mapped to its index in the tokenizer vocabulary. For this, we will use the tokenizer included with BERT. The following lines will download the BERT model for Finnish and other necessary files, and then load the pre-trained BertTokenizer.

To illustrate the tokenizeation process, let's look at one sentence, split into tokens and map it to token ids :

```python
from transformers import BertTokenizer

PRE_TRAINED_MODEL_NAME = "TurkuNLP/bert-base-finnish-cased-v1"

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

sentence = "toimii uskomattoman hyvin tarkasti paljon parempi kuin useimmat muut vastaavat löytää hyvin lähetykset"
print('Original: ', sentence)
print('Tokenized: ', tokenizer.tokenize(sentences[0]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
```
> Original:  toimii uskomattoman hyvin tarkasti paljon parempi kuin > useimmat muut vastaavat löytää hyvin lähetykset 

> Tokenized:  ['toimii', 'uskomattoman', 'hyvin', 'tarkasti', 'paljon', > 'parempi', 'kuin', 'useimmat', 'muut', 'vastaavat', 'löytää', 'hyvin', 'lähetykset']

> Token IDs:  [2181, 16600, 658, 8048, 640, 2634, 341, 10534, 531, 9038, >3200, 658, 42109]

BERT requires adding some special tokens, we add [CLS] to the beginning of each sentnce to specify that we are doing classification, [SEP] to mark the end of each sentence and [PAD] for padding. As sentences length may vary, we use max_length to pad and tunncate all sentences to a single constant length. All this is done using `encode_plus` function. 


```python
sentences = df.content_cleaned.values
labels = df.sentiment_label.values


input_ids = []
attention_masks = []

for sentence in sentences:

    encoded_dict = tokenizer.encode_plus(
                        sentence,                      .
                        add_special_tokens = True,
                        max_length = 256,
                        pad_to_max_length = True,
                        return_attention_mask = True, 
                        return_tensors = 'pt',
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)
```

Next, we divide our dataset to training (80%), validation (10%) and testing set (10%).

```python
from torch.utils.data import TensorDataset, random_split

dataset = TensorDataset(input_ids, attention_masks, labels)


train_size = int(0.8 * len(dataset))
val_size = (len(dataset) - train_size) // 2

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,val_size])

print('Training samples', train_size)
print('Validation samples', val_size)
print('Test samples', val_size)
```

For better memory usage, we will use pytorch's DataLoader class :

```python

BATCH_SIZE = 32

train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = BATCH_SIZE 
        )

validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), .
            batch_size = BATCH_SIZE 
        )

test_dataloader = DataLoader(
            test_dataset, 
            sampler = SequentialSampler(test_dataset), 
            batch_size = BATCH_SIZE 
        )
```


# Training The Classifier

We will finetune the pre-trained BERT to adapt it to our sentiment classification task. We will be using BertForSequenceClassification interface provided by huggingface pytorch implementation and designed for our task. It's basically a normal BERT model with an extra single linear layer on the top for classification.

The documentation for `from_pretrained` can be found [here](https://huggingface.co/transformers/v2.2.0/main_classes/model.html#transformers.PreTrainedModel.from_pretrained), with the additional parameters defined [here](https://huggingface.co/transformers/v2.2.0/main_classes/configuration.html#transformers.PretrainedConfig).

```python
from transformers import BertForSequenceClassification,  BertConfig

NUM_LABELS = 3

model = BertForSequenceClassification.from_pretrained(
    PRE_TRAINED_MODEL_NAME, 
    num_labels = NUM_LABELS, 
    output_attentions = False, 
    output_hidden_states = False, 
)

model.cuda()
```

We’ll use the AdamW optimizer provided by Hugging Face (as used in BERT paper), and a linear scheduler with no warmup steps. For the hyperparameters, we will follow BERT authors recommendations.

```python
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8 
                )

EPOCHS = 4
TOTAL_STEPS = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = TOTAL_STEPS)

```




```python
import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128 and the following article : https://mccormickml.com/2019/07/22/BERT-fine-tuning/

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    torch.save(model.state_dict(), f'bert_model_state_{epoch_i}.bin')
    print(FileLink(f'./bert_model_state_{epoch_i}.bin'))
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
```

 WIP, Notebook to be organized ... 