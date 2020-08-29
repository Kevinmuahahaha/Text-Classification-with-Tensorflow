import pickle
import jieba
import os

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

print("Tokenizing Chinese strings with jieba...")

tokens_dict = None
tokens_index = 90
tokens_file_name = "customized_tokens_dict"

def get_token( words ):
    global tokens_dict
    global tokens_index

    if tokens_dict == None:
        if os.path.isfile(tokens_file_name+".pkl"):
            print("Local tokens found. Using it...")
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


stored_dict = load_obj("data_dict")
jieba.load_userdict("./customized_dict.txt")
for item in stored_dict:
    seg_list = jieba.lcut(stored_dict[item]['text'])    # using default .cut()
    for i in range(len(seg_list)):                      # all chars are in lower case
        seg_list[i] = seg_list[i].lower()
        seg_list[i] = get_token( seg_list[i] )          # tokenizing all words into numbers
    stored_dict[item]['text'] = seg_list                # refresh the dict

if not os.path.isfile(tokens_file_name+".pkl"):
    save_obj(tokens_dict, tokens_file_name)             # if local token dict doesn't exist, then create one


save_obj(stored_dict, "data_dict")
print("Saved dictionary to: ", "data_dict.pkl")         # this should've been the last step
                                                        # following codes save the data into json format


for item in stored_dict:                                # list to string, for later use
    _int_text = stored_dict[item]['text']
    _str_text = [str(integer) for integer in _int_text]
    stored_dict[item]['text'] = ' '.join(_str_text)


json_dict = {}
# { class_num:xxx, text_list:[...,] }
for item in stored_dict:                                # prettify our data, for later use
    if len(stored_dict[item]['text']) < 50:
        continue                                        # skip extra-short articles(could be images and such)
    class_nums_list = stored_dict[item]['tags']
    for one_class in class_nums_list:
        try:
            json_dict[one_class]                        # if such class exist, append the article to it
            json_dict[one_class].append( stored_dict[item]['text'] )
        except:
            json_dict[one_class] = list()
            json_dict[one_class].append(  stored_dict[item]['text'] )

print("Saving data into json format.")
output_json = open("test.json","w")
index_count = 1
dict_len = len(json_dict)

output_json.write("{")
for one_class in json_dict:
    output_json.write("\"" + str(one_class) + "\":[\n")
    
    #texts
    list_len = len(json_dict[one_class])
    article_index = 1
    for article in json_dict[one_class]:
        output_json.write("\"" + str(article) + "\"")    # write the text body, surrounded with ""
        if article_index != list_len:                 # if it's not the last item, then add a trailing comma
            output_json.write(",\n")
        else:
            output_json.write("\n")
        article_index += 1

    if index_count == dict_len:                     
        output_json.write("]\n")
    else:
        output_json.write("],\n")                     # if that's not the last class, then add a trailing comma
    index_count += 1
output_json.write("}")

output_json.close()


class_count = 0
for one_class in json_dict:
    class_count += 1
print("Class count: ", class_count)
