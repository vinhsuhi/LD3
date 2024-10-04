import random
import torch
import json 

class RandomNumberIterator:
    def __init__(self, model, scale, batch_size, n_classes=1000):
        self.model = model
        self.scale = scale
        self.batch_size = batch_size
        self.n_classes = n_classes
    
    def __iter__(self):
        return self
    
    def __next__(self):
        label = torch.LongTensor([random.randint(0, self.n_classes - 1) for _ in range(self.batch_size)]).to(self.model.device)
        conditioning = self.model.get_learned_conditioning({self.model.cond_stage_key: label})
        if self.scale != 1.0:
            conditioned_unconditioning = conditioning.clone()
        else:
            conditioned_unconditioning = None

        return conditioning, conditioned_unconditioning
    
class UniformNumberIterator:
    def __init__(self, model, scale, batch_size, num_samples_per_class, n_classes=1000):
        self.model = model
        self.scale = scale
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.n_classes = n_classes
        self.current_value = 0 
        self.current_num_cls_sample = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        # Prepare the batch with the current value
        batch = [self.current_value] * self.batch_size
        self.current_num_cls_sample += self.batch_size
        if self.current_num_cls_sample >= self.num_samples_per_class:
            # Update the current value, cycling through 0 to 1000
            self.current_value = (self.current_value + 1) % self.n_classes
            self.current_num_cls_sample = 0 

        label = torch.LongTensor(batch).to(self.model.device)
        conditioning = self.model.get_learned_conditioning({self.model.cond_stage_key: label})
        if self.scale != 1.0:
            conditioned_unconditioning = self.model.get_learned_conditioning({self.model.cond_stage_key: torch.LongTensor([self.n_classes] * self.batch_size).to(self.model.device)})
        else:
            conditioned_unconditioning = None

        return conditioning, conditioned_unconditioning
    
class TextFileIterator:
    def __init__(self, model, scale, file_path, batch_size, max_prompts=None):
        self.model = model
        self.scale = scale
        self.unconditional_conditioning = self.model.get_learned_conditioning([""])

        self.file_path = file_path
        self.batch_size = batch_size
        self.max_prompts = max_prompts
        self.prompt_index = 0
        self.prompts = self._load_prompts()

    def __iter__(self):
        return self

    def __next__(self):
        if self.prompt_index >= len(self.prompts):
            raise StopIteration

        batch_prompts = self.prompts[self.prompt_index:self.prompt_index + self.batch_size]
        self.prompt_index += self.batch_size

        conditioning = self.model.get_learned_conditioning(batch_prompts)
        conditioned_unconditioning = self.unconditional_conditioning.repeat(self.batch_size, 1, 1)
        return conditioning, conditioned_unconditioning

        
    def _load_prompts(self):
        try:
            if self.file_path.endswith('json'):
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    mscoco_data = json.load(file)
                    prompts = [_['caption'] for _ in mscoco_data['annotations']]
                    if self.max_prompts is not None:
                        prompts = prompts[:self.max_prompts]
                    return prompts
            else:
                return [prompt.strip() for prompt in open(self.file_path)][:self.max_prompts]
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return []
        except IOError as e:
            print(f"Error reading file {self.file_path}: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {self.file_path}: {e}")
            return []


class HPSv2Iterator:
    def __init__(self, model, scale, file_path, batch_size, max_prompts=None):
        self.model = model
        self.scale = scale
        self.unconditional_conditioning = self.model.get_learned_conditioning([""])

        self.batch_size = batch_size
        self.max_prompts = max_prompts
        self.prompt_index = 0
        self.prompts = self._load_prompts(subset=file_path)

    def __iter__(self):
        return self

    def __next__(self):
        if self.prompt_index >= len(self.prompts):
            raise StopIteration

        batch_prompts = self.prompts[self.prompt_index:self.prompt_index + self.batch_size]
        self.prompt_index += self.batch_size

        conditioning = self.model.get_learned_conditioning(batch_prompts)
        conditioned_unconditioning = self.unconditional_conditioning.repeat(self.batch_size, 1, 1)
        return conditioning, conditioned_unconditioning

    def _load_prompts(self, subset):
        import hpsv2
        return hpsv2.benchmark_prompts(subset) 