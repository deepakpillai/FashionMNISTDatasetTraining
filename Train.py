import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score


class Train():
    def __init__(self):
        pass

    def train_model(self, model, training_data, testing_data, number_of_epoch):
        loss_fn = nn.CrossEntropyLoss()
        optim_fn = torch.optim.SGD(params=model.parameters(), lr=0.1)

        total_training_loss = 0

        for epoch in tqdm(range(0, number_of_epoch)):
            print("\n\n-------- \n")
            print(f"Epoch {epoch}")
            for batch_number, (features, labels) in enumerate(training_data):
                model.train()
                logits = model(features)
                pred = logits.argmax(dim=1)
                loss = loss_fn(logits, labels)
                optim_fn.zero_grad()
                loss.backward()
                optim_fn.step()
                total_training_loss += loss

                if batch_number % 400 == 0:
                    print(f"Looked through {batch_number} batches")
                    print(f"Average training loss {total_training_loss / len(training_data)}")
                    print(f"Accuracy score {accuracy_score(pred, labels)}")

            for test_batch_number, (test_features, test_labels) in enumerate(testing_data):
                with torch.inference_mode():
                    model.eval()
                    test_logits = model(test_features)
                    test_pred = test_logits.argmax(dim=1)
                    if test_batch_number % 200 == 0:
                        print(f"Testing accuracy score {accuracy_score(test_pred, test_labels)}")
