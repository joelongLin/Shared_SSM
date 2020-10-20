# -*-coding:utf-8-*-
import time
import os
import json

def reload_config(FLAGS):
    # If we are reloading a models, overwrite the flags
    if FLAGS.reload_model is not '':
        with open(os.path.join(FLAGS.reload_model, 'hyperparameter.json')) as data_file:
            config_dict = json.load(data_file)
            #必须加进去，否则会丢失 reload_model
            config_dict['reload_model'] =  FLAGS.reload_model
        for key, value in config_dict.items():
            attr_remove = ['gpu', 'run_name', 'logs_dir', 'n_steps_gen', 'display_step', 'generate_step']
            # attr_remove = ['gpu', 'n_steps_gen', 'reload_model', 'display_step', 'generate_step']
            if key not in attr_remove:
                FLAGS.__setattr__(key, value)
    return FLAGS