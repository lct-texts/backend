import torch
from transformers import AutoTokenizer, AutoModel
from .models import Message, Coords, Conversation
from annoy import AnnoyIndex
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import DocumentSearchPipeline
from haystack import Document
from nltk.corpus import stopwords
from string import punctuation
from pymystem3 import Mystem
from collections import Counter
from sklearn.manifold import TSNE
import numpy as np
from decimal import Decimal
from sklearn.feature_extraction.text import TfidfVectorizer


stop_themes = [
    'вопрос клиента связанный с отказом от использования продуктов банка',
    'жалобы',
    'просроченная задолженность',
    'мошенничество, утеря/кража карты'
]

mystem = Mystem()
russian_stopwords = stopwords.words("russian")


model = None
tokenizer = None
index = None
bm25_pipeline = None
stop_themes_index = None


def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    return text

def load_bm25():
    global bm25_pipeline
    if not bm25_pipeline:
        store = InMemoryDocumentStore(use_bm25=True)
        store.write_documents(
            list(
                map(
                    lambda x: 
                        Document(
                            answer=x.message, 
                            content=x.lemmatization_message,
                            meta={
                                'id': x.id,
                                'name': x.message
                            }
                        ), 
                        Message.objects.filter(author='user', lemmatization_message__isnull=False)
                    )
                )
        )
        bm25 = BM25Retriever(document_store=store)
        bm25_pipeline = DocumentSearchPipeline(bm25)


def get_model():
    global model
    if not model:
        model = AutoModel.from_pretrained("ai-forever/ruBert-large")

def get_tokenizer():
    global tokenizer
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-large")

def get_index():
    global index
    if not index:
        index = AnnoyIndex(1024, 'angular')
        index.load('index.annoy')


def model_loaded(func):
    def wrapper(*args, **kwargs):
        get_model()
        get_tokenizer()
        return func(*args, **kwargs)
    return wrapper


def index_loaded(func):
    def wrapper(*args, **kwargs):
        get_index()
        return func(*args, **kwargs)
    return wrapper

def bm25_loaded(func):
    def wrapper(*args, **kwargs):
        load_bm25()
        return func(*args, **kwargs)
    return wrapper


@model_loaded
def calculate_embeddings(query):
    encoded_input = tokenizer([query], padding=True, truncation=True, max_length=64, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)

    return embeddings[0]

@index_loaded
def bert_candidates(embs, num):
    global index
    ids = index.get_nns_by_vector(embs, num)
    ans = []
    for i in ids:
        ans.append(Message.objects.get(id=i))
    return ans

@bm25_loaded
def bm25_candidates(query, num):
    res = bm25_pipeline.run(query=query, params={'Retriever': {'top_k': num}})
    return list(
        map(
            lambda x: Message.objects.get(id=x.meta['id']),
            res['documents']
        )
    )


def load_stop_themes():
    global stop_themes_index
    if stop_themes_index == None:
        stop_themes_index = AnnoyIndex(1024, 'angular')
        stop_themes_index.load('stop_themes.annoy')


def stop_themes_loaded(func):
    def wrapper(*args, **kwargs):
        load_stop_themes()
        return func(*args, **kwargs)
    return wrapper


@stop_themes_loaded
def run_stop_theme_nearest(msgs):
    global stop_themes_index, stop_themes
    nearest = []
    for msg in msgs:
        msg_embeddings = calculate_embeddings(msg)
        nearest.append(stop_themes_index.get_nns_by_vector(msg_embeddings, 1)[0])
    print(Counter(nearest).most_common(1)[0][0])
    return stop_themes[Counter(nearest).most_common(1)[0][0]]


def build_stop_themes():
    stop_themes = [
        'вопрос клиента связанный с отказом от использования продуктов банка',
        'жалобы',
        'просроченная задолженность',
        'мошенничество, утеря/кража карты'
    ]
    stop_themes_embedding = list(map(calculate_embeddings, stop_themes))
    stop_themes_index = AnnoyIndex(1024, 'angular')
    for i, embedding in enumerate(stop_themes_embedding):
        stop_themes_index.add_item(i, embedding)
    stop_themes_index.build(20)
    stop_themes_index.save('stop_themes.annoy')

    Coords.objects.filter(type='stop_theme').delete()
    msgs = list(filter(lambda x: len(x.message.split()) > 5, Message.objects.filter(conversation__sentiment='negative', author='user')))
    msgs_embeddings = list(map(lambda x: calculate_embeddings(x.message), msgs))
    labels = []
    for embedding in msgs_embeddings:
        labels.append(stop_themes[stop_themes_index.get_nns_by_vector(embedding, 1)[0]])
    
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=3, n_iter=10000).fit_transform(np.array(msgs_embeddings))
    
    for i, msg in enumerate(msgs):
        Coords.objects.create(
            message=msg,
            type='stop_theme',
            x=Decimal(str(X_embedded[i, 0])),
            y=Decimal(str(X_embedded[i, 1])),
            label=labels[i]
        )



def build_coords(items):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(list(map(preprocess_text, items)))
    Coords.objects.all().delete()
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=3, n_iter=10000).fit_transform(X.toarray())
    msg_used = set()
    for i, item in enumerate(items):
        if len(item.split()) < 3: continue
        if len(msg_used | {item}) == len(msg_used): continue
        msgs = Message.objects.filter(message=item)
        for msg in msgs:
            Coords.objects.create(
                x=Decimal(str(X_embedded[i, 0])),
                y=Decimal(str(X_embedded[i, 1])),
                message=msg,
                label=msg.conversation.sentiment,
                type='sentiment'
            )
        msg_used.add(item)
    build_stop_themes()


@model_loaded
def index_data():
    global index
    max_len_messages = []
    for conversation in Conversation.objects.all():
        msgs = list(conversation.message_set.filter(author='user'))
        max_msg = ''
        for msg in msgs:
            if msg.author == 'user':
                if len(max_msg) < len(msg.message):
                    max_msg = msg.message
        max_len_messages.append(max_msg)
    
    max_items = list(map(lambda x: x, max_len_messages))
    build_coords(max_items)
    
    items = list(map(lambda x: {'name': x.message, 'id': x.id, 'embeddings': calculate_embeddings(x.message)}, Message.objects.filter(author='user')))
    for item in Message.objects.filter(author='user'):
        item.lemmatization_message = preprocess_text(item.message)
        item.save()
    index = AnnoyIndex(1024, 'angular')
    for item in items:
        index.add_item(item['id'], item['embeddings'])
    index.build(20)
    index.save('index.annoy')


@model_loaded
def calculate_jaccard_score(query, result):
    def jaccard(a, b):
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    prep_query = tokenizer([preprocess_text(query)], padding=True, truncation=True, max_length=64, return_tensors='pt')[0].ids[1:-1]
    prep_result = tokenizer([preprocess_text(result)], padding=True, truncation=True, max_length=64, return_tensors='pt')[0].ids[1:-1]
    return jaccard(set(prep_query), set(prep_result))


@model_loaded
@index_loaded
@bm25_loaded
def search(query):
    n = 5
    embs = calculate_embeddings(query)
    bert_nearest = bert_candidates(embs, n)
    bm25_nearest = bm25_candidates(preprocess_text(query), n)
    candidates = bert_nearest + bm25_nearest
    filtered_candidates = list()
    already_mapped_candidates = set()
    for item in candidates:
        if len(already_mapped_candidates | {item.id}) != len(already_mapped_candidates):
            filtered_candidates.append(item)
            already_mapped_candidates.add(item.id)
    
    sorted_candidates = list(
        reversed(
            list(
                map(
                    lambda x: x,
                    sorted(
                        map(
                            lambda x: {
                                'item': x,
                                'score': calculate_jaccard_score(query, x.message)
                            },
                            filtered_candidates
                        ),
                        key=lambda x: x['score']
                    )
                )
            )
        )
    )
    return sorted_candidates


def calculate_metrics_based_on_search(candidates):
    mapping_values = {
        'positive': 5,
        'neutral': 0,
        'negative': -5
    }
    inv_values = {v: k for k, v in mapping_values.items()}

    sorts = list(
        map(
            lambda x: mapping_values[x['item'].conversation.sentiment],
            filter(
                lambda y: y['score'] > 0,
                candidates
            )
        )
    )
    sentiment = Counter(sorts).most_common(1)[0][0]
    return {
        'score': sum(sorts) / len(sorts), 
        'sentiment': inv_values[sentiment]
    }
