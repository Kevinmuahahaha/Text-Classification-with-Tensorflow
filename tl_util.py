import os
import pickle
import re
import jieba

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def get_token( words, tokens_dict, tokens_index ):
    tokens_file_name = "customized_tokens_dict"
    if tokens_dict == None:
        if os.path.isfile(tokens_file_name+".pkl"):     # using local dict
            tokens_dict = load_obj(tokens_file_name)
        else:
            tokens_dict = dict()

    try:
        tokens_dict[words]
        return tokens_dict[words]
    except:
        tokens_dict[words] = tokens_index
        tokens_index += 1
        return tokens_dict[words]



unwanted_strings = [
        '本站部分内容均来自互联网,如不慎侵害的您的权益,请告知,我们将尽快删除.',
        'Part of the information in our website is from the internet.',
        'If by any chance it violates your rights,',
        'we will delete it upon notification as soon as possible.',
        'Thank you for cooperation.'
        '\(', '\)',
        '（','）',
        '立陶宛','李银河','C先生'
        ]


def clean_text(text):
    new_text = text
    for rgx_match in unwanted_strings:
        new_text = re.sub(rgx_match, '', new_text)
    return new_text


tags_dict = load_obj("tl_tag_dictionary")
def num_to_name( num ):
    return tags_dict[num]
