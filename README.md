# multiclass_text_classfier
This project is aimed to classify non mutex human errors from Nuclear power plant incident reports on a sentece level. Besides indentifying errors with a pin point accuracy, This could also help with analysing the relationships between the errors.


## General Guidelines:
~make sure the class distribution is not very skewed
<br/> ~the hyper parameters should be consistent for train and prediction
<br/> ~Use the same binarizer and tokenizer for train and prediction
<br/> ~If using the predict function in web server. it should be called in a seperate process (python multiprocessing) due to the way memory is handled
