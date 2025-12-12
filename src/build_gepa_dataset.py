import pandas
import dspy
import random
import os
import json
import pickle

class BuildGEPADataset:
    def __init__(self, max_train_per_context: int = 5, max_val_per_context: int = 20, context_to_exclude: list[str] = ["natural image", "medical image", "synthetic scene"], output_dir="data", save_dataset=False, random_seed: int = 42):
        self.max_train_per_context = max_train_per_context
        self.max_val_per_context = max_val_per_context
        self.context_to_exclude = context_to_exclude
        self.output_dir = output_dir
        self.save_dataset = save_dataset
        self.random_seed = random_seed
    
    def load_test(self, file_name):
        if os.path.exists(file_name):
            df = pandas.read_parquet(file_name)
        else:
            print(f"Data not found at {file_name}")
            return []

        data = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Ensure image path is correct relative to root if it starts with 'images/'
            if 'image' in row_dict and isinstance(row_dict['image'], str):
                if row_dict['image'].startswith('images/') and not row_dict['image'].startswith('data/'):
                    row_dict['image'] = os.path.join(self.output_dir, row_dict['image'])
                elif row_dict['image'].startswith('images/') and not os.path.exists(row_dict['image']):
                     # Fallback if output_dir is different or path structure differs
                     row_dict['image'] = os.path.join("data", row_dict['image'])
                     
                # Wrap with dspy.Image
                row_dict['image'] = dspy.Image(row_dict['image'])

            # Create dspy Example
            example = dspy.Example(**row_dict).with_inputs('question', 'choices', 'image')
            data.append(example)
        return data

    def build(self):
        # Set random seed
        random.seed(self.random_seed)

        data_path = os.path.join(self.output_dir, "testmini.parquet")
        if os.path.exists(data_path):
            df = pandas.read_parquet(data_path)
        else:
            print(f"Data not found at {data_path}") 
            return [], []
        
        # Group by context
        context_groups = {}
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            
            metadata = row_dict.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    pass
            
            if isinstance(metadata, dict):
                context = metadata.get('context')
                if context in self.context_to_exclude:
                    continue
                
                if context not in context_groups:
                    context_groups[context] = []
                
                # Ensure image path is correct
                if 'image' in row_dict and isinstance(row_dict['image'], str):
                     if row_dict['image'].startswith('images/') and not row_dict['image'].startswith('data/'):
                        row_dict['image'] = os.path.join(self.output_dir, row_dict['image'])
                     elif row_dict['image'].startswith('images/') and not os.path.exists(row_dict['image']):
                         row_dict['image'] = os.path.join("data", row_dict['image'])
                         
                     # Wrap with dspy.Image
                     row_dict['image'] = dspy.Image(row_dict['image'])

                example = dspy.Example(**row_dict).with_inputs('question', 'choices', 'image')
                context_groups[context].append(example)
        
        train_set = []
        val_set = []
        
        for context, examples in context_groups.items():
            random.shuffle(examples)
            
            # Take max_train_per_context
            train_subset = examples[:self.max_train_per_context]
            train_set.extend(train_subset)
            
            # Take max_val_per_context from the remaining? Or just next batch?
            # Assuming disjoint: train first, then val.
            remaining = examples[self.max_train_per_context:]
            val_subset = remaining[:self.max_val_per_context]
            val_set.extend(val_subset)
            
        random.shuffle(train_set)
        random.shuffle(val_set)
        
        if self.save_dataset:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            
            with open(os.path.join(self.output_dir, "train_set.pkl"), "wb") as f:
                pickle.dump(train_set, f)
            with open(os.path.join(self.output_dir, "val_set.pkl"), "wb") as f:
                pickle.dump(val_set, f)
                
        return train_set, val_set

if __name__ == "__main__":
    # Example usage
    # builder = BuildGEPADataset(
    #     max_train_per_context=5, 
    #     max_val_per_context=2, 
    #     context_to_exclude=["natural image"], 
    #     output_dir="data",
    #     save_dataset=False,
    #     random_seed=42
    # )
    builder = BuildGEPADataset()
    
    # Build train and validation sets
    print("Building dataset...")
    train, val = builder.build()
    print(f"Train size: {len(train)}")
    print(f"Val size: {len(val)}")
    
    # Load test set
    print("Loading test set...")
    test_file = os.path.join("data", "testmini.parquet")
    test = builder.load_test(test_file)
    print(f"Test size: {len(test)}")
