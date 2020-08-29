# raw text --> read line by line
    # 1. clean up each line
    # 2. text --jieba--> list of text
    # 3. list --customized tokenization--> input string
    # 3. string --predictor--> output

from tl_util import clean_text, get_token, load_obj
import jieba

tokens_file_name = "customized_tokens_dict"
tokens_dict = load_obj(tokens_file_name)
tokens_index = 90

def file_to_input_string( path_to_file ):
    line_list = []

    f = open(path_to_file,"r")
    for line in f:
        line = clean_text(line)
        if len(line) <= 0:
            pass
        line_list.append( line )

    string_for_jieba = ' '.join( line_list )

    jieba.load_userdict("./customized_dict.txt")
    seg_list = jieba.lcut( string_for_jieba )
    for i in range(len(seg_list)):
        seg_list[i] = seg_list[i].lower()
        seg_list[i] = get_token( seg_list[i], tokens_dict, tokens_index )       ## to numbers

    seg_string_list = [str(integer) for integer in seg_list]
    return ' '.join(seg_string_list)
