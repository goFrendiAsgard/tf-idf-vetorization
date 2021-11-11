from typing import List, Mapping
import math
import re

def documents_to_document_index(documents: Mapping[str, str]) -> Mapping[str, Mapping[str, int]]:
    document_index: Mapping[str, int] = {}
    for key, sentence in documents.items():
        document_index[key] = {}
        words = sentence_to_words(sentence)
        for word in words:
            if word not in document_index[key]:
                document_index[key][word] = 0
            document_index[key][word] += 1
    return document_index

def get_reversed_document_index(document_index: Mapping[str, Mapping[str, int]]) -> Mapping[str, Mapping[str, int]]:
    reversed_document_index: Mapping[str, int] = {}
    for doc_key, term_frequency in document_index.items():
        for term_key, frequency in term_frequency.items():
            if term_key not in reversed_document_index:
                reversed_document_index[term_key] = {}
            reversed_document_index[term_key][doc_key] = frequency
    return reversed_document_index

def sentence_to_words(sentence: str) -> List[str]:
    return sentence.split(' ')

def boolean_retrieval_and(reversed_document_index: Mapping[str, Mapping[str, int]], term_1, term_2) -> List[str]:
    set_1 = reversed_document_index[term_1]
    set_2 = reversed_document_index[term_2]
    result = []
    for doc_1 in set_1:
        for doc_2 in set_2:
            if doc_2 == doc_1:
                result.append(doc_1)
            break
    return result

def boolean_retrieval_or(reversed_document_index: Mapping[str, Mapping[str, int]], term_1, term_2) -> List[str]:
    set_1 = reversed_document_index[term_1]
    set_2 = reversed_document_index[term_2]
    result = []
    for doc_1 in set_1:
        result.append(doc_1)
    for doc_2 in set_2:
        if doc_2 not in result:
            result.append(doc_2)
    return result

def normalize_documents(documents: Mapping[str, str]) -> Mapping[str, str]: 
    normal_documents = {}
    for key, sentence in documents.items():
        lower_sentence = sentence.lower()
        alphanum_sentence = re.sub(r'[^a-z0-9 ]', '', lower_sentence)
        normal_documents[key] = re.sub(r' +', ' ', alphanum_sentence)
    return normal_documents

def get_tf(reversed_document_index: Mapping[str, Mapping[str, int]], term: str, doc: str) -> int:
    if term in reversed_document_index and doc in reversed_document_index[term]:
        return reversed_document_index[term][doc] 
    return 0

def get_df(reversed_document_index: Mapping[str, Mapping[str, int]], term: str) -> int:
    if term in reversed_document_index:
        return len(reversed_document_index[term].keys())
    return 0

def get_idf(reversed_document_index: Mapping[str, Mapping[str, int]], term: str, doc_count: int) -> float:
    df = get_df(reversed_document_index, term)
    return math.log(doc_count/df)

def get_tf_idf(reversed_document_index: Mapping[str, Mapping[str, int]], term: str, doc: str, doc_count: int) -> float:
    tf = get_tf(reversed_document_index, term, doc)
    idf = get_idf(reversed_document_index, term, doc_count)
    return tf * idf

def vectorize(reversed_document_index: Mapping[str, Mapping[str, int]], doc: str, terms: List[str], doc_count: int) -> List[float]:
    vector = []
    for term in terms:
        tfidf = get_tf_idf(reversed_document_index, term, doc, doc_count)
        vector.append(tfidf)
    return vector

def get_dot_product(v1: List[float], v2: List[float]) -> float:
    result = 0.0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

def get_vector_length(v1: List[float]) -> float:
    result = 0.0
    for i in range(len(v1)):
        result += v1[i] ** 2
    result = result ** 0.5
    return result

def get_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = get_dot_product(v1, v2)
    v1_length = get_vector_length(v1)
    v2_length = get_vector_length(v2)
    return dot_product / (v1_length * v2_length)


my_documents: Mapping[str, str] = {
    'd1' : 'Lorem Ipsum is simply dummy text of the printing and typesetting industry',
    'd2': 'Lorem Ipsum has been the industry standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book',
    'd3': 'It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s',
}
my_doc_count = len(my_documents.keys())
my_normal_documents = normalize_documents(my_documents)
print('case folding + normalization')
print(my_normal_documents)
my_document_index: Mapping[str, Mapping[str, int]] = documents_to_document_index(my_normal_documents)
print('indexing')
print(my_document_index)
my_reversed_document_index: Mapping[str, Mapping[str, int]] = get_reversed_document_index(my_document_index)
print('reverse indexing')
print(my_reversed_document_index)
print('lorem AND typesetting')
print(boolean_retrieval_and(my_reversed_document_index, 'lorem', 'typesetting'))
print('lorem OR typesetting')
print(boolean_retrieval_or(my_reversed_document_index, 'lorem', 'typesetting'))
print('TF "it" on doc 3')
print(get_tf(my_reversed_document_index, 'it', 'd3')) # 2
print('TF "it" on doc d1')
print(get_tf(my_reversed_document_index, 'it', 'd1')) # 0
print('DF "it"')
print(get_df(my_reversed_document_index, 'it')) # 2 --> d2 and d3
print('IDF "it"')
print(get_idf(my_reversed_document_index, 'it', my_doc_count))
print('TF.IDF "it" on doc 1')
print (get_tf_idf(my_reversed_document_index, 'it', 'd1', my_doc_count))
print('TF.IDF "it" on doc 3')
print (get_tf_idf(my_reversed_document_index, 'it', 'd3', my_doc_count))
print('TF.IDF "centuries" on doc 3')
print (get_tf_idf(my_reversed_document_index, 'centuries', 'd3', my_doc_count))

my_terms = my_reversed_document_index.keys()
print('vector d1')
v1 = vectorize(my_reversed_document_index, 'd1', my_terms, my_doc_count)
print(v1)
print('vector d2')
v2 = vectorize(my_reversed_document_index, 'd2', my_terms, my_doc_count)
print(v2)

print('cosine similarity v1 and v2')
print(get_cosine_similarity(v1, v2))
