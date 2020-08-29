# Text-Classification-with-Tensorflow
Auto tagging based on article content. Using proprietary data provided by transtives.net, which won't be provided.

This is a project made for non-profit transgender organization Translives based in Mainland China.

Translives aims to provide unbiased up-to-date information for transgenders across China.

# Prepare your data
Data fetched from database should look like: ( dictionary )
```
	{id:xxx, { title:xxx, text:xxx, tags:[ ...] } }
```

With pickel, this dictionary should be saved to "data_dict.pkl"

with tl_util.py you may do:
```
	save_obj( your_dict, "data_dict" )
```

Tags should be saved to a seperate dictionary "tl_tag_dictionary" also using pickel.


# Define your own dictionary(Chinese)
"customized_dict.txt" is used to assist jieba module, so that jieba knows which words should be grouped together(instead of taking them apart).


# Reprocess data(tokenizing etc.)
"data_reprocessor.py" processes pickel files, using jieba and so on, then perform tokenization.
The output will look like this(in json format):

	{
		"Class 1" : [
			"Text 1",
			"Text 2",
		],
		"Class 2" : [],
		"Class 3" : []
	}


# Training your model
"tensor_backend.py" training backend.

# Prepare your data for prediction
"predict_data_reprocessing.py" processes file for predicting.

# Predict a file using trained model.
"predictor.py" where you actually use the predicting function.


