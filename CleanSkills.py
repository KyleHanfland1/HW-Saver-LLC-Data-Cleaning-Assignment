#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pattern.vector import Document, KNN
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import OneClassSVM
import time


#load the dataframes and remove unnecessary punctuation and lowercase strings in the dataframes keeping periods for .Net and .Net experience and spaces for word seperation
example_hard_skills_df = pd.read_csv('Example_Technical_Skills.csv')
skills_to_clean_df = pd.read_csv('Raw_Skills_Dataset.csv')
example_soft_skills_df = pd.read_csv('Example_Soft_Skills.csv')
example_hard_skills_df["Technology Skills"] = example_hard_skills_df["Technology Skills"].str.replace('[{}]'.format("""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~"""), '')
skills_to_clean_df["RAW DATA"] = skills_to_clean_df["RAW DATA"].str.replace('[{}]'.format("""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~"""), '')
combined_df = pd.concat([example_hard_skills_df, skills_to_clean_df.rename(columns={'RAW DATA':'Technology Skills'})], axis=0)


#@param combined_df dataframe object consisting of the dataframe for example technical skills concatenated with the raw skill data
#@param n the number of entries at the start that correspond to only example technical skills
#@param num_iter the number of iterations that the OCSVM will undergo
#ex. combined_df.head(n) == example_hard_skills_df
#@return hard_skills_df dataframe holding the skill strings that the OCSVM classified as a positive case (predict == 1)
#@return rejected_skills_df dataframe holding the skill strings that the OCSVM classified as a negative case (predict == -1)
def cleanOCSVM(combined_df, n, num_iter=1000):
    #no need to lowercase as it has already happened, remove english articles and non essential words, 
    tfidf = TfidfVectorizer(lowercase=False, sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    train_tfidf = tfidf.fit_transform(combined_df['Technology Skills']).toarray()
    test_tfidf = tfidf.transform(combined_df['Technology Skills'].tail(combined_df.size-n))
    print("Beginning Training OCSVM")
    start = time.time()
    model = OneClassSVM(max_iter=num_iter, nu=.01).fit(train_tfidf) #generally we want a smaller nu because training error should be low with no outliers in data
    print("Training the OCSVM took " +str(time.time()-start) + " seconds")
    print("Beginning predictions on model")
    start = time.time()
    predictions = model.predict(test_tfidf.toarray())
    print("Predictions took " +str(time.time()-start)+ " seconds")
    duplicate_check_set = set()
    hard_skills_df = []
    rejected_skills_df = []
    for index, i in enumerate(predictions):
        entry = skills_to_clean_df.loc[index, "RAW DATA"]
        if(entry == ""):continue
        if(entry not in duplicate_check_set):
            duplicate_check_set.add(entry)
            if(i==1):
                hard_skills_df.append(entry)
            else:
                rejected_skills_df.append(entry)
    hard_skills_df = pd.DataFrame(hard_skills_df, columns = ['Cleaned Hard Skills'])
    rejected_skills_df = pd.DataFrame(rejected_skills_df, columns = ['Cleaned Soft Skills'])
    return hard_skills_df, rejected_skills_df


#@param example_hard_skills_df dataframe object holding example hard skills
#@param example_soft_skills_df dataframe object holding example soft skills
#@param skills_to_clean_df dataframe object holding all the skills to classify
#@return hard_skills_df dataframe holding the skill strings that the KNN classified as a hard skill 
#@return rejected_skills_df dataframe holding the skill strings that the KNN classified as a soft skill 
def cleanKNN(example_hard_skills_df, example_soft_skills_df, skills_to_clean_df):
    hard_skills_df = [] #starts as a list for cheaper append
    rejected_skills_df = []
    check_duplicate_set = set()
    knn = KNN()
    print("Begin Loading the Data into KNN model")
    start = time.time()
    for index, row in example_hard_skills_df.iterrows():
        #row['Technology Skills'] is the string entry at index index
        hard_skill_string = row['Technology Skills']
        hard_tag_count = hard_skill_string.split()
        hard_tag_count_dict = {x:hard_tag_count.count(x) for x in hard_tag_count}
        knn.train(hard_tag_count_dict, type="HardSkill")
    for index, row in example_soft_skills_df.iterrows():
        soft_skill_string = row['Soft Skills']
        soft_tag_count = soft_skill_string.split()
        soft_tag_count_dict = {x:soft_tag_count.count(x) for x in soft_tag_count}
        knn.train(soft_tag_count_dict, type="SoftSkill")
    print("Finished Loading the Data into KNN model in " + str(time.time()-start) + " seconds.")
    print("Beginning Predictions for KNN model")
    start = time.time()
    for index, row in skills_to_clean_df.iterrows():
        unknown_skill_string = row['RAW DATA']
        if(unknown_skill_string == ""):continue
        if(unknown_skill_string not in check_duplicate_set): #reduce as many redundant cases of classify
            check_duplicate_set.add(unknown_skill_string)
            if(knn.classify(unknown_skill_string) == "HardSkill"):
                hard_skills_df.append(unknown_skill_string)
            else:
                rejected_skills_df.append(unknown_skill_string)
    print("Finished Predictions and sorting results in " + str(time.time()-start) + " seconds.")
    hard_skills_df = pd.DataFrame(hard_skills_df, columns = ['Cleaned Hard Skills'])
    rejected_skills_df = pd.DataFrame(rejected_skills_df, columns = ['Cleaned Soft Skills'])
    return hard_skills_df, rejected_skills_df


#cleanup and exporting results to .csv files 
OCSVM_hard_skills_df, OCSVM_rejected_skills_df = cleanOCSVM(combined_df, example_hard_skills_df.size, num_iter=10000)
KNN_hard_skills_df, KNN_soft_skills_df = cleanKNN(example_hard_skills_df, example_soft_skills_df, skills_to_clean_df)
OCSVM_hard_skills_df.to_csv('OCSVM_Cleaned_Hard_Skills.csv')
OCSVM_rejected_skills_df.to_csv('OCSVM_Cleaned_Rejected_Skills.csv')
KNN_hard_skills_df.to_csv('KNN_Cleaned_Hard_Skills.csv')
KNN_soft_skills_df.to_csv('KNN_Cleaned_Soft_Skills.csv')


