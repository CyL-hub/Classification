import os

digits = [i for i in range(500, 5001, 500)]
faces = [i for i in range(45, 452, 45)]
trainRatio = [float(i/10) for i in range(1,11)]

for ratio, d, f in zip(trainRatio, digits, faces):
    print(f"---Training @ {ratio}---")
    # os.system(f'py -2 dataClassifier.py -c knn -t {d}')
    # os.system(f"py -2 dataClassifier.py -c knn -t {f} -d faces")
    os.system(f"py -2 dataClassifier.py -c perceptron -t {d}")
    os.system(f"py -2 dataClassifier.py -c perceptron -t {f} -d faces")


    
# Change face validation from test to val
# Re run on face data