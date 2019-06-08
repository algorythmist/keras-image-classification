import os
from horse_or_human_loader import prepare_dataset

if __name__ == '__main__':
    directory = prepare_dataset()

    # Directory with our training horse pictures
    train_horse_dir = os.path.join(directory, 'horses')

    # Directory with our training human pictures
    train_human_dir = os.path.join(directory, 'humans')

    print('total training horse images:', len(os.listdir(train_horse_dir)))
    print('total training human images:', len(os.listdir(train_human_dir)))