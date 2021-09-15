from NERDA.datasets import get_conll_data, download_conll_data 

from NERDA.models import NERDA


# get data as dictionary type

download_conll_data()
training = get_conll_data('train')
validation = get_conll_data('valid')

# available NER tags for the task 

tag_scheme = [
'B-PER',
'I-PER',
'B-ORG',
'I-ORG',
'B-LOC',
'I-LOC',
'B-MISC',
'I-MISC'
]

# tranformer being fine tuned. (this is from hugging face)

transformer = 'bert-base-multilingual-uncased'

# a selection of basic hyperparameters for the network as well as for the model training itself.

# hyperparameters for network
dropout = 0.1
# hyperparameters for training
training_hyperparameters = {
                            'epochs' : 4,
                            'warmup_steps' : 500,
                            'train_batch_size': 13,
                            'learning_rate': 0.0001
                            }

# model config. using the NERDA model interface

# from NERDA.models import NERDA

model = NERDA(
            dataset_training = training,
            dataset_validation = validation,
            tag_scheme = tag_scheme, 
            tag_outside = 'O',
            transformer = transformer,
            dropout = dropout,
            hyperparameters = training_hyperparameters
        )

# to train model. ..TRAIN   
model.train()


# to evaluate model.    ..EVALUATE

test = get_conll_data('test')
model.evaluate_performance(test)



# to predict using the model. (Identifying named entities in a text)    ..PREDICT
model.predict_text('Cristiano Ronaldo plays for Juventus FC')