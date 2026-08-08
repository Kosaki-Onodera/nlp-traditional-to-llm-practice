import numpy as np  
import pandas as pd
import os
from gensim.models import Word2Vec
from Bags_of_Words import review_to_words
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

#******获取词簇袋的方法******
def create_bag_of_centroids( words, word_centroid_map ):
    #簇从0开始算，所以要+1
    num_centroids = max( word_centroid_map.values() ) + 1

    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

    for word in words:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
            
    return bag_of_centroids

if __name__ == '__main__':
    model = Word2Vec.load("300features_40minwords_10context")
    type(model.wv.vectors)
    model.wv.vectors.shape

    #******读取数据******
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,delimiter="\t",quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )

    #******处理数据******
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append( review_to_words(review))        
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append( review_to_words(review))


    #******K-mean算法******
    word_vectors = model.wv.vectors
    num_clusters = word_vectors.shape[0] / 5
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )

    #******创建单词到簇的映射******
    word_centroid_map = dict(zip( model.wv.index2word, idx ))

    #******创建训练集和测试集的簇袋******
    train_centroids = np.zeros( (train["review"].size, num_clusters), \
    dtype="float32" )

    counter = 0
    for review in clean_train_reviews:
       train_centroids[counter] = create_bag_of_centroids( review, \
          word_centroid_map )
       counter += 1

    #******创建测试集的簇袋******
    test_centroids = np.zeros(( test["review"].size, num_clusters), \
       dtype="float32" )

    counter = 0
    for review in clean_test_reviews:
       test_centroids[counter] = create_bag_of_centroids( review, \
          word_centroid_map )
       counter += 1

    #*****训练随机森林分类器******
    forest = RandomForestClassifier(n_estimators = 100)

    print ("用带标签的训练数据拟合随机森林模型")
    forest = forest.fit(train_centroids,train["sentiment"])
    result = forest.predict(test_centroids)

    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
    output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )
    
    