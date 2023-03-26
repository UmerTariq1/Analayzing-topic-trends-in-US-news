# Analayzing-topic-trends-in-US-news

This project was for my masters course project for Computaitonal Linguistics taught by Prof. Koller.
In this project I do the data analysis of US news category dataset released in 2021 september. I use NER and LDA for my analysis. NER to understand what personalities have been discussed in this data while LDA to see what topic are discussed in this dataset. 

## Directory Structure
.
├── main.ipynb (notebook where the analysis is done)
├── lda.py (custom lda implementation of LDA using gibbs sampling)
├── readme.txt
├── report.pdf
├── images (output images folder of data analysis)
    ├── ner output (output visualization images for analysis done by NER) 
    ├── lda output images (output visualization images for analysis done by LDA)
├── ner output (output folder for ner pickle files)
    ├── ner output (output visualization images for analysis done by NER) 
    ├── lda output images (output visualization images for analysis done by LDA)
├── lda output (output folder for lda models pickle files. Also has pyLDAvis visualization in html form)
    ├── lda trained model pkl files
    ├── lda_vis_output (folder containing lda html topic modeling visualizations using pyLDAvis)
├── data (folder containing source data)
    ├── News_Category_Dataset_v3.json


## Versions
nltk - 3.7
wordcloud - 1.8.2.2
spacy - 3.5.1
spacy-transformers - 1.2.2
pyLDAvis - 3.4.0

you can use requirements.txt to create a conda envrionment

## Runtime
Its a notebook so every cell can be run separately. those cells which will take a lot of runtime have this mentioned at the top of the cell.
