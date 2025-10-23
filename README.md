# SeleCom

This is the official implementation for the Full Paper: **Rethinking Soft Compression in Retrieval-Augmented Generation: A Query-Conditioned Selector Perspective**

<img src="figures/introduction.png" width=50%></img>


## Introduction of this repository

This repository provides the implementation, data pipeline, and pretrained checkpoints of SeleCom. It contains the following contents:

```
.
├── checkpoint                --> (Folder saving checkpoints of Stage 1 and Stage 2 models)
├── data                      --> (Folder containing raw corpus and processed training data)
├── dataprocess               --> (Functions for corpus cleaning, QA generation, and data synthesis)
├── figures                   --> (Figures used in paper visualization and README)
├── log                       --> (Logs for training and evaluation)
├── main                      --> (Main entry file; training and evaluation are run through this file)
├── model                     --> (Core implementation of the Selector, Projector, and Generator modules)
└── util                      --> (Utility functions for metrics, configuration, and data loading)
```
**Note !!!** The checkpoint/ and data/ folders are too large to be hosted directly on GitHub.
They will be uploaded to Hugging Face Hub after the review process is complete.
(Links will be added here once anonymity constraints are lifted.)

## Paper Intro

**Main Pipeline** SeleCom replaces traditional full document compression with a query-conditioned selection mechanism. It includes a Selector that extracts query-relevant information from retrieved documents, a Projector that aligns compressed embeddings with the generator’s space, and a Generator that produces grounded answers using both the query and these embeddings. The framework is trained in two stages: Stage 1 trains the Selector and Projector for accurate information selection, and Stage 2 fine-tunes the Generator to utilize the compressed embeddings effectively.

<img src="figures/pipeline.png" width=70%></img>

**Data Construction** The training corpus is built through an LLM-assisted pipeline starting from the Wikipedia dump (~33M documents). Non-informative documents are filtered out, and LLMs are used to generate, verify, and grade QA pairs by difficulty. The resulting dataset, containing about 14 million (query, document, answer) triples, supports curriculum learning and enables robust training of the query-conditioned Selector.

<img src="figures/dataprocess.png" width=70%></img>

## Comming soon...
## Run SeleCom

### Environmental Requirement


### Reproduce the performance of paper version


### Run on your datasets or your models 

## Example

## Else







  