# NewsEncoder 설정
class Config:
    gpu_num = 0
    seed = 28
    use_batch = True
    batch_size = 150
    hop = 3
    interval_minutes = 2160

    num_words = 1 + 330899  
    word_embedding_dim = 100  
    num_categories = 34
    num_categories_for_NewsEncoder = 16
    num_subcategories_for_NewsEncoder = 80
    num_filters = 100  
    query_vector_dim = 200  
    window_size = 3
    dropout_probability = 0.2
    
    method = 'multihead_self_attention' 
    
    head_num = 20
    head_dim = 15
    dataset_lang = 'norwegian'
    category_emb_dim = 100
    subcategory_emb_dim = 100
    attention_hidden_dim = 100
    
    no_category = False       
    unique_category = False  
    adjust_score = True       

