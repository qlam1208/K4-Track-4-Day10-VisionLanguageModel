"""Chat examples and padding; image tensors are concatenated, never text-padded."""
from pathlib import Path
from PIL import Image
from student_vlm.labels import objects_to_answer, prompt_for


def open_image(record, data_root):
    with Image.open(Path(data_root)/record['image_path']) as source:
        image = source.convert('RGB')
    if image.size != (record['width'], record['height']):
        raise ValueError(f"Image dimensions disagree for {record['image_id']}")
    return image


def user_message(image, coord='norm1000'):
    return {'role':'user','content':[{'type':'image','image':image},
                                    {'type':'text','text':prompt_for(coord)}]}


def encode_example(processor, image, objects, coord='norm1000', expected_vision_tokens=220):
    import torch
    user = user_message(image, coord)
    answer = objects_to_answer(objects, coord=coord, width=image.width, height=image.height)
    common = dict(tokenize=True, return_dict=True, return_tensors='pt')
    prefix = processor.apply_chat_template([user], add_generation_prompt=True, **common)
    full = dict(processor.apply_chat_template(
        [user, {'role':'assistant','content':answer}], add_generation_prompt=False, **common))
    n = prefix['input_ids'].shape[1]
    if not torch.equal(full['input_ids'][:, :n], prefix['input_ids']):
        raise ValueError('Chat generation prompt is not a prefix of the training sequence')
    image_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    image_tokens = int((full['input_ids'] == image_id).sum())
    if expected_vision_tokens is not None and image_tokens != expected_vision_tokens:
        raise ValueError(f'Expected {expected_vision_tokens} vision tokens, got {image_tokens}')
    labels = full['input_ids'].clone()
    labels[:, :n] = -100
    if not (labels != -100).any():
        raise ValueError('No supervised answer tokens')
    full['labels'] = labels
    full.pop('token_type_ids', None)  # Transformers Qwen3-VL example uses mm_token_type_ids instead.
    return full


def collate_examples(examples, pad_token_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    if not examples:
        raise ValueError('Cannot collate an empty batch')
    result = {}
    for key, pad in [('input_ids',pad_token_id),('attention_mask',0),('labels',-100),
                     ('mm_token_type_ids',0)]:
        present = [key in e for e in examples]
        if any(present) and not all(present):
            raise ValueError(f'Inconsistent batch field: {key}')
        if all(present):
            result[key] = pad_sequence([e[key].squeeze(0) for e in examples],
                                      batch_first=True, padding_value=pad)
    for key in ('pixel_values','image_grid_thw'):
        if not all(key in e for e in examples):
            raise ValueError(f'Missing vision field: {key}')
        result[key] = torch.cat([e[key] for e in examples], dim=0)
    return result


class RoadDataset:
    def __init__(self, records, processor, data_root='data', coord='norm1000'):
        self.records, self.processor, self.data_root, self.coord = records, processor, data_root, coord

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return encode_example(self.processor, open_image(r,self.data_root), r['objects'],self.coord)
