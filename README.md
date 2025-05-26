# machine learning notes 

i asked claude and chatgpt to generate me a course plan which i can follow to learn llms. the aim is to start from machine learning fundamentals and build upto large language models. the topics which i aim to cover (not exhaustive and will be updated over time):

## build core understanding of statistical learning, optimization, and supervised/unsupervised paradigms.
### core concepts & mathematics
* ml paradigms:
    * supervised vs unsupervised learning
    * regression vs classification
    * overfitting, underfitting, bias-variance tradeoff
    * training/validation/test splits
* mathematics refresher:
    * linear algebra: vectors, matrices, eigenvalues, svd
    * calculus: derivatives, gradients, chain rule
    * probability & stats: distributions, bayes theorem, confidence intervals
### algorithms & evaluation
* algorithms:
    * linear/logistic regression
    * decision trees, random forests, svms
    * k-means, pca
* evaluation & tuning:
    * metrics: accuracy, precision, recall, f1, roc
    * cross-validation
    * hyperparameter tuning (grid, random search)
* python tooling:
    * numpy, pandas, matplotlib
    * scikit-learn basics
    * jupyter/vscode  
### project 1: 
* build full ml pipeline (classification/regression) using scikit-learn.

## understand neural networks, backpropagation, optimization, and deep architectures.
### neural network fundamentals
* perceptron, mlps
    * forward/backward propagation
    * activation: relu, sigmoid, tanh
* loss functions: mse, cross-entropy
* optimization: sgd, adam, rmsprop
* regularization: dropout, batch norm, weight decay
### frameworks & training pipelines
* tooling:
    * pytorch/tensorflow: tensors, autograd, dataloaders
    * training loop construction
* cnns:
    * convolution, pooling, feature maps
    * architectures: lenet, alexnet, resnet
    * transfer learning, data augmentation
### rnns and sequence modeling
* vanilla rnns, vanishing gradients
* lstm, gru, bidirectional rnns
* applications: text classification, sequence-to-sequence
* intro to attention (prelude to transformers)  
### project 2:
* cnn on cifar-10
* lstm on text sentiment classification (imdb or sst-2)

## internalize attention mechanisms, transformer architecture, and classic nlp pipelines.
### traditional nlp
* text preprocessing: tokenization, lemmatization, stemming
* representations: bow, tf-idf
* word embeddings: word2vec, glove
* pos tagging, named entity recognition
### attention & seq2seq
* encoder-decoder models
* additive vs multiplicative attention
* self-attention: q/k/v, positional encoding
* multi-head attention
### transformer architecture
* "attention is all you need" breakdown
    * transformer block: self-attention + ffn + layernorm + residual
    * training: masking, positional encoding, teacher forcing
* model variants:
    * bert: mlm, nsp
    * gpt: causal lm
    * t5/bart: sequence-to-sequence tasks  
### project 3:  
* implement basic transformer from scratch (translation task).
* fine-tune bert/gpt-2 on classification dataset.

## acquire implementation-level understanding of llm scaling, training, finetuning, and inference.
### pretraining pipeline
* dataset construction: common crawl, c4, deduplication
* tokenization: bpe, wordpiece, sentencepiece
* objective: causal lm (next token prediction)
* curriculum learning, mixed precision, gradient accumulation
### scaling & optimization
* scaling laws: chinchilla, performance vs parameters
* memory efficiency: activation checkpointing, zero, fsdp
* distributed training: data, model, pipeline parallelism
### finetuning & alignment
* supervised finetuning, instruction tuning
* lora, adapters, bitfit (parameter-efficient finetuning)
* rlhf: reward models, ppo, constitutional ai
### inference & deployment
* sampling: greedy, beam, top-k, top-p
* quantization: int8, int4
* distillation & speculative decoding
* serving: huggingface + accelerate, vllm, onnx, tensorrt
* api design, safety filtering, monitoring  
### project 4:
* fine-tune llama/flan-t5 using lora on a domain-specific dataset
* deploy with basic inference api

### capstone projects
1. pretrain tiny llm  
    * train mini gpt-like model on wikipedia subset  
    * evaluate perplexity, loss curve convergence  
2. rag system  
    * build vector store (faiss)  
    * integrate with llm using langchain  
    * add memory and conversational context  
3. model compression  
    * apply quantization + pruning  
    * benchmark tradeoffs in performance vs efficiency  
    * deploy optimized model  
4. evaluation framework  
    * benchmark llms across multiple tasks  
    * metrics: bleu, rouge, f1, bias tests  
    * analyze fairness, toxicity, hallucination rate
