from torchwnn.classifiers import Wisard
from torchwnn.encoding import Thermometer

from datahandler import load_data
import torch
import numpy as np



if __name__ == '__main__':
    data_points, labels = load_data("complex9")
    print(np.unique(labels, return_counts=True))
    entry_size = data_points.shape[1]
    tuple_size = 64
    bits_encoding = 16
    encoding = Thermometer(bits_encoding).fit(data_points)
    X_bin = encoding.binarize(data_points).flatten(start_dim=1)

    model = Wisard(entry_size=entry_size, tuple_size=tuple_size, n_classes=len(np.unique(labels)))
    with torch.no_grad():
        model.fit(input = torch.Tensor(X_bin), target=torch.Tensor(labels).int())
        prediction = model.predict(X_bin)
        score = 1
        for i in range(len(prediction)):
            print(prediction[i], labels[i])
            if prediction[i] == labels[i]:
                score += 1
        print(score/len(prediction))
