# download and unzip the dataset from kaggle
[group('kaggle')]
kaggle: kaggle-download kaggle-unzip

# download the dataset from kaggle
[group('kaggle')]
kaggle-download:
    curl -L -o $(pwd)/data/default-of-credit-card-clients-dataset.zip \
        https://www.kaggle.com/api/v1/datasets/download/uciml/default-of-credit-card-clients-dataset

# unzip the dataset
[group('kaggle')]
kaggle-unzip:
    unzip -o $(pwd)/data/default-of-credit-card-clients-dataset.zip -d $(pwd)/data