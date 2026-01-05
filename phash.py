from PIL import Image 
import imagehash
import os
import pickle


embed_path = 'embed_index_result/embeds/'
index_path = 'embed_index_result/index/'
result_path = 'embed_index_result/results/'

def phash(image_path):
    return imagehash.phash(Image.open(image_path))


with open(f'{embed_path}index_resnet50_embeddings.pkl', 'rb') as f:
    index_data = pickle.load(f)
index_meta_data = index_data['meta_data']


index_hashes = []
for m in index_meta_data:
    h = phash(m['path'])
    index_hashes.append({
        'index_class': m['class'],
        'index_instance_id': m['instance_id'],
        'hash': h,
        'path': m['path']
    })


with open(f'{embed_path}queries_resnet50_embeddings.pkl', 'rb') as f:
    query_data = pickle.load(f)
query_meta_data = query_data['meta_data']


k = 12

results = []
for query in query_meta_data:

    query_hash = phash(query['path'])

    res = []
    
    for index in index_hashes:

        res.append({
            # - is overloeads in imagehash to compute the hamming distance 
            # between hashes (integer respresting the number of differing bits )
            'score':  query_hash - index['hash'], 

            'index_class': index['index_class'],
            'index_instance_id': index['index_instance_id'],
            'path' : index['path']
        })

    # sort by the hamming distance. the smaller the value, the more similar the index is to the query    
    res.sort(key= lambda x:x['score'])
    
    # select top-k most similar 
    res = res[:k]
    
    results.append({'query_class': query['class'],  'query_instance_id':  query['instance_id'], 'query_path': query['path'], 
                        'matches': res})


#save results
with open (f'{result_path}phash_results.pkl', 'wb') as f:
    pickle.dump(results,f)


'''Evaluatiing PHashing'''

k_list = [1,3,5,10,11]

for k in k_list:
    tp_instance = 0 # true positives for instance level recall
    tp_class = 0 # true positives for instance level recall  
    total = 0 # total number of queries

    for r in results:
        total += 1

        true_query_class = r['query_class']
        true_query_instance_id = r['query_instance_id']

        topk_classes = [ res['index_class'] for res in r['matches'][:k]]
        topk_instances = [res['index_instance_id'] for res in r['matches'][:k]]
        

        '''if true_query_class in topk_classes:
            tp_class += 1
        
        if true_query_instance_id in topk_instances:
            tp_instance += 1'''
        
        is_counted_already = False

        # we use the looped nested logic because an instance is only a correct prediction if it is from the right class. 
        # This helps prevent counting  images that have the same instance_id but are from different classes
        for claSS, instance  in zip(topk_classes, topk_instances):
            # prevents recounting classes that appear move than once top-k results 
            if true_query_class == claSS:
                if is_counted_already == False:
                    tp_class +=1
                    is_counted_already = True
                if true_query_instance_id == instance:
                    tp_instance += 1
        
    instance_lvl_recall_at_k = tp_instance / total
    class_lvl_recall_at_k = tp_class /( total)

    print(f'Phash Instance level Recall@{k} = {instance_lvl_recall_at_k:.4f}')
    print(f'Phash Class level Recall@{k} = {class_lvl_recall_at_k:.4f}', end='\n\n')





