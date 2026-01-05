import pickle
import faiss
import numpy as np

embed_path = 'embed_index_result/embeds/'
index_path = 'embed_index_result/index/'
result_path = 'embed_index_result/results/'

# load  index embeddings 
with open(f'{embed_path}index_resnet50_embeddings.pkl', 'rb') as f:
    index_data = pickle.load(f)
index_embeddings = index_data['embeddings']
index_meta_data = index_data['meta_data']

d = index_embeddings.shape[1]

# build index ( flat index since we aredealing with small dataset)
index = faiss.IndexFlatIP(d) #inner product 
index.add(index_embeddings) # add vectors 

# save index to disk
faiss.write_index(index, f'{index_path}faiss_resnet50_IndexFlatIP.index')

# load queries embedding 
with open(f'{embed_path}queries_resnet50_embeddings.pkl', 'rb') as f:
    query_data = pickle.load(f)
query_embeddings = query_data['embeddings']
query_meta_data = query_data['meta_data']


k = 12
S_S, I = index.search(query_embeddings, k = k) # S_S: cosine similarity scores, I:indices


results = []
for i, (s_s,indices) in enumerate(zip(S_S,I)):
    qinfo = query_meta_data[i]
    res = []
    
    #load most similar in res
    for score, indx in zip(s_s, indices):
        ''' note might have to come back and add img clas and id '''
        res.append({'score': float(score), 'index_id':indx, 'index_class': index_meta_data[indx]['class'], 
                    'index_instance_id': index_meta_data[indx]['instance_id'], 'index_path': index_meta_data[indx]['path'] }) 
    

    results.append({'query_class': qinfo['class'],  'query_instance_id':  qinfo['instance_id'], 'query_path': qinfo['path'], 
                    'matches': res})

#save results
with open (f'{result_path}faiss_resnet50_results.pkl', 'wb') as f:
    pickle.dump(results,f)
