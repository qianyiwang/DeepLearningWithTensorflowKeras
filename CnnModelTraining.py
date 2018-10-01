from TrainingDataProcessing import CreateTrainingData

createTraining = CreateTrainingData('data/train/', 'data/test/', 50, ['cat','dog'])
if not createTraining.isCreated():
    createTraining.createTrainingData()
    createTraining.saveData()
    print('Trainind data has been created and saved')
else:
    print('Trainind data already exist')
