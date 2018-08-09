# Djent_Learning


University of Glasgow Master Project: Machine learning approaches to sound effect synthesis
	
	Set up:
		1. python 3.6 (must be 64bit version)
		
		2. keras 2.2.2
		
		3. tensorflow 1.9.0
		
		4. matplotlib 2.2.2
		
		5. numpy 1.15.0
	
	The way to use:
		1. Put the related clean guitar audio files and drive audio files in directories "codes/data/train/input" and "codes/data/train/input"
		
		2. Run "codes/djent_learning.py" to create a model and train it. The model will be saved in "codes/data/model"
		
		3. Put test clean guitar audio files in "codes/data/test/input"
		
		4. Run "codes/test_with_audio.py", the graphs of test outputs will be shown. The output audio files will be saved in "codes/test/output" after closing the graph windows.
