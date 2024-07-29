from wolof_translate import *


def recuperate_datasets(
    char_p: float,
    word_p: float,
    max_len: int,
    end_mark: int,
    tokenizer: T5TokenizerFast,
    corpus_1: str = "french",
    corpus_2: str = "wolof",
    train_file: str = "data/extractions/new_data/train_set.csv",
    test_file: str = "data/extractions/new_data/test_file.csv",
    augmenter=partial(nac.RandomCharAug, action="swap"),
):

    # Let us recuperate the end_mark adding option
    if end_mark == 1:
        # Create augmentation to add on French sentences
        fr_augmentation_1 = TransformerSequences(
            augmenter(aug_char_p=fr_char_p, aug_word_p=fr_word_p, aug_word_max=max_len),
            remove_mark_space,
            delete_guillemet_space,
            add_mark_space,
        )

        fr_augmentation_2 = TransformerSequences(
            remove_mark_space, delete_guillemet_space, add_mark_space
        )

    else:

        if end_mark == 2:

            end_mark_fn = partial(add_end_mark, end_mark_to_remove="!", replace=True)

        elif end_mark == 3:

            end_mark_fn = partial(add_end_mark)

        elif end_mark == 4:

            end_mark_fn = partial(add_end_mark, end_mark_to_remove="!")

        else:

            raise ValueError(f"No end mark number {end_mark}")

        # Create augmentation to add on French sentences
        fr_augmentation_1 = TransformerSequences(
            augmenter(aug_char_p=fr_char_p, aug_word_p=fr_word_p, aug_word_max=max_len),
            remove_mark_space,
            delete_guillemet_space,
            add_mark_space,
            end_mark_fn,
        )

        fr_augmentation_2 = TransformerSequences(
            remove_mark_space, delete_guillemet_space, add_mark_space, end_mark_fn
        )

    # Recuperate the train dataset
    train_dataset_aug = SentenceDataset(
        train_file,
        tokenizer,
        truncation=False,
        cp1_transformer=fr_augmentation_1,
        cp2_transformer=fr_augmentation_2,
        corpus_1=corpus_1,
        corpus_2=corpus_2,
    )

    # Recuperate the valid dataset
    valid_dataset = SentenceDataset(
        test_file,
        tokenizer,
        cp1_transformer=fr_augmentation_2,
        cp2_transformer=fr_augmentation_2,
        corpus_1=corpus_1,
        corpus_2=corpus_2,
        truncation=False,
    )

    # Return the datasets
    return train_dataset_aug, valid_dataset
