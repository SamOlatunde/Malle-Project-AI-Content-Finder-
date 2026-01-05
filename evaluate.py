import pickle 
import numpy as np

embed_path = 'embed_index_result/embeds/'
index_path = 'embed_index_result/index/'
result_path = 'embed_index_result/results/'

with open(f'{result_path}faiss_resnet50_results.pkl', 'rb') as f:
    results = pickle.load(f)

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

    print(f'resnet50 Instance level Recall@{k} = {instance_lvl_recall_at_k:.4f}')
    print(f'resnet50 Class level Recall@{k} = {class_lvl_recall_at_k:.4f}', end='\n\n')





