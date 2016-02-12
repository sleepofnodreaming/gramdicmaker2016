
from grdicmaker import DictionaryCollector, DataTransformer
from test_data_readers import LangTestData

if __name__ == '__main__':
    transfomer = DataTransformer(
        "training_sets/kz/training_data_full.json",
        "training_sets/kz/paradigm_lengths.json",
        "category_description/kazakh.json"
    )
    matrix_kz_full, targets_kz_full = transfomer.get_training_data_matrix(normalize=True)
    
    