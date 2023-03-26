# Analayzing-topic-trends-in-US-news

This project was for my masters course project for Computaitonal Linguistics taught by Prof. Koller.
In this project I do the data analysis of US news category dataset released in 2021 september. I use NER and LDA for my analysis. NER to understand what personalities have been discussed in this data while LDA to see what topic are discussed in this dataset. 

Project structure:
  - lda.py (custom lda implementation of LDA using gibbs sampling)
  - main.ipynb (notebook where the analysis is done)
  - images (output images folder of data analysis)
    |
     -- ner output (output visualization images for analysis done by NER)
     -- lda output (output visualization images for analysis done by LDA)
  - ner output (output folder for ner pickle files)
  - lda output (output folder for lda models pickle files)
      |
      -- lda_vis_output (folder containing lda html topic modeling visualizations using pyLDAvis)
  - data (folder containing source data)
     |
     -- News_Category_Dataset_v3.json
