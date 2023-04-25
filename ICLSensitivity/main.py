
# task data:
# template: {'instruction': str, 'verbalizers': list(str)}
# auxiliary_instructions: list({'instruction': str, 'verbalizers': list(str)})
# main_inst, auxiliary_insts [optional]: prompt as a string with <text> and <verbalizer> as placeholders
# examples {input, label}, label as index of 0, ..., c
# few-shot example as one-file, test examples as another file
# specify args: model_name, label_bias_adjustment (none, cc, pc)
# return: predictions with confidences (automatically give confidences from various calibration methods?)

# data files:
# train.json: list[dict({'text': str, 'label': int})]
# test.json: list[dict({'text': str})]

# command args:
# instruction: str
# auxiliary_instructions: optional, list[str]
# model_name, load_model_dir, load_tokenizer_path, etc.
# label_bias_adjustment

