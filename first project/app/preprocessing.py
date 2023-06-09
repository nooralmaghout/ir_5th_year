import nltk
import ir_datasets
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(nltk.corpus.stopwords.words('english'))
tfidf_vect = TfidfVectorizer()

def preprocess(dataset_name):
    print(f"hi {dataset_name}")
    dataset = pd.read_csv( fr"C:\Users\pc\.ir_datasets\{dataset_name}\collection.tsv" , encoding='utf-8',sep='\t', header=None ) 
    dataset.columns = ('key', 'document' )
    dataset['cleaned_document'] = dataset['document'].apply(lambda x: [word.lower() for word in str(x).split() if word.lower() not in stop_words])
    print(dataset.head())
    dataset['lowered_document'] = dataset['cleaned_document'].apply(lambda x: " ".join(x))
    print(dataset.head())
    # TFIDF
    x = tfidf_vect.fit_transform(dataset['lowered_document'])
    print(x.shape)
    return(x)

# preprocess("antique")

def query_preprocess(query):
    porter= PorterStemmer()
    tokens =query.split()
    stemmed_tokens =[porter.stem(token.lower()) for token in tokens if token.lower() not in stop_words]
    print(stemmed_tokens)
    return ' '.join(stemmed_tokens)

# query_preprocess('how can we get concentration onsomething?')

#Query Preprocess
def handle_query():
    # print(f"hi {dataset_name}")
    dataset = ir_datasets.load("antique/test")
    queries =[]
    queries_id =[]
    for query in dataset.queries_iter():
        queries_id.append(query[0])
        queries.append(query[1])
    # print(queries)
    #query Tfidf
    query_vector = tfidf_vect.transform([query_preprocess('how can we get concentration onsomething?')])
    print(query_vector.shape)
    return(query_vector)

# handle_query()

#Cosine Similarity
def calculate_cos_similarity():
    cosine_similarities =[]
    tfidf_matrices = preprocess("antique")
    tfidf_query = handle_query()
    for tfidf_matrix in  tfidf_matrices:
        cosine = cosine_similarity(tfidf_query,tfidf_matrix).flatten()
        cosine_similarities.append(cosine)
    #Most Similar
    most_similar_documents = sorted(range(len(cosine_similarities)), key= lambda i : cosine_similarities[i],reverse=True)
    print(most_similar_documents)

# calculate_cos_similarity()

def handle_qrels():
    # print(f"hi {dataset_name}")
    # ir_datasets export antique/test qrels --format tsv
    qrels_df=pd.read_csv( fr"C:\Users\pc\.ir_datasets\antique\test\qrels.TSV",encoding='utf-8', sep=' ',header =None,)
    
    #   dataset.columns = ('key', 'document' )
    qrels_df.columns = ('query_id' , 'iteration' , 'doc_id' ,'relevance')
    print(qrels_df.head())
    print(len(qrels_df))
# handle_qrels()