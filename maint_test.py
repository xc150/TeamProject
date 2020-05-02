from model.core import predict

instance = [
            50, ' Federal-gov', 83311, ' Doctorate', 13,
            ' Married-civ-spouse', ' Exec-managerial', ' Husband',
            ' White', ' Male', 0, 0, 10.0, ' United-States'
        ]
instances = [instance]

print(predict(instances)[0])