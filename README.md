# Search-Engine
 Our Proposal for the "Modern Search Engines" Group project

## Creators:
 - Lukas Weber
 - Dana Rapp
 - Simon ...
 - Maximilian Jaques

## Search Engine Structure
1. Crawler: Crawls the web for documents starting from the frontier
    - Create Frontier
       - https://uni-tuebingen.de/
       - https://www.tuebingen.de/
    - Be able to stop in between. With Done and Todo list
    - Consider Access control (robots.txt)
    - Duplicate detection (exact and near duplicates)
      - [ ] Simhash
      - [ ] Autoencoders
    - Relevanz von Websites checken
3. Indexing: Saving the crawled content. Optimizing speed and performance in finding relevant documents for a search query so the search engine does not have to scan every document in the corpus & Matching query and document
     - Index selection mit PageRank? (Prüfen ob Document in den Index sollte)
     - Contentprüfung
     - Text representation
       - if necessary: Lower-Casing, Removing Stop words (exchanging . for <end> token)
       - Topic modelling (LDA)
         OR
       - Word Embeddings
         OR additionally
       - Contextual embedding (Transformer)
         - [ ] Transformer Memory as a Differentiable Search Index
5. Retrieval & Ranking: Process of reading the index and returning the desired results
    - Conv-KNRM
    - Col-Bert
6. Search Engine Interface (Communicates with 3. and 4.)
    - Autocomplete
7. ((What about Fairness and Bias? (Part of the Slides about the project phase)))

## TODOs
- Finish Crawler
- Choose Frontier
- Indexing Method
- Matching
- Ranking
- Interface

## Requirements
- Python 3.xx
- Pytorch
- NLTK


## Keep in Mind:
- Pipeline (processing) applied to all documents needs to be applied to the queries, too
