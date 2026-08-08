import numpy as np  
import pandas as pd
import os
from gensim.models import Word2Vec
from Bags_of_Words import review_to_words
from sklearn.ensemble import RandomForestClassifier

def makeFeatureVec(words, model, num_features):
    
    # 初始化全 0 向量，用来累加所有单词向量。
    featureVec = np.zeros((num_features,),dtype="float32")
    
    #统计这篇文本中，出现在词表中的有效单词数量。
    nwords = 0.
    
    # model.index2word 是模型全部词汇列表；转成set集合，查询word in set速度很快。
    index2word_set = set(model.index2word)
    
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    
    counter = 0.
    
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
    
    counter = counter + 1.
    return reviewFeatureVecs

if __name__ == '__main__':
    model = Word2Vec.load("300features_40minwords_10context")
    type(model.wv.vectors)
    model.wv.vectors.shape

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,delimiter="\t",quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append( review_to_words( review, \
        remove_stopwords=True ))

    trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

    print ("Creating average feature vecs for test reviews")
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append( review_to_words( review, \
        remove_stopwords=True ))

    testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

    forest = RandomForestClassifier( n_estimators = 100 )

    print ("Fitting a random forest to labeled training data...")
    forest = forest.fit( trainDataVecs, train["sentiment"] )

    result = forest.predict( testDataVecs )

    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )