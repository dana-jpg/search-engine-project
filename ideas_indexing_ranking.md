Datenstrukturen Indexing

1. InvertedIndex dictionary key: term e.g. tuebingen,wilhemstrasse value: [] with all document ids containing this word, [] position in document #Entries take the form [docID, [pos1, pos2, ...]], ordered e.g 
square_range = [[3, [65, 90]], [8, [67, 94]], [12, [3]], [19, [18, 81, 1881]], [23, [63]]]
function: find relevant documents-> []
2. Index dictionary key:docid incremential int  value: url

Ranking

1. doc representaion 1 for TF-IDF, all out of the inverted index, calculate for all the relevant documents -> list with 100 best documents with score
2. ?

Workflow:

Offline:

Input: text of the html document (convert html to plain text!! different version) \
Preprocessing: lower, stopword removal, stemming *Simon* \
Indexing: add url and increment id \
Inverted index: efficient way *Max* \

Online:
Input: Query
Preprocessing function ['food', 'drinks']
Use inverted index to get relevant document ids [ids], terms
Calculate TF-IDF score for each document

Maybe testing