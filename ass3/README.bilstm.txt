STEVE GUTFREUND – 342873791
YOEL JASNER – 204380992

Additional options
- bilstmTrain.py
	we need two additional parameters:
		1) devFile, a file with tagged data on which the model is NOT training on but just checking how good the model performs
		2) task, this should be 'ner' or 'pos', the model distinguish between the two types in matter of accuracies (as in assignment 2)
		
	a correct command would be:
	>> python bilstmTrain.py a ner/train ner_model_a ner/dev ner

- bilstmTag.py
	we need two additional parameters:
		1) parameters.json, a json file containing parameters needed to build the model, this file is created by running bilstmTrain.py
		2) outputFile, the file in which the predictions should be written to
		
	a correct command would be:
	>> python bilstmTag.py a ner_model_a ner_parameters.json test4.ner
