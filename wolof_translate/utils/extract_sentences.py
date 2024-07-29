from nlp_project import *
import pickle


class LengthError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CommandError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ExtractRelatedSentences:

    nb_process = 1

    indices = {}

    def __init__(
        self,
        corpora: pd.DataFrame,
        corpus_1: str = "french_corpus",
        corpus_2: str = "wolof_corpus",
    ):

        self.corpora = corpora

        self.corpus_1 = corpus_1

        self.corpus_2 = corpus_2

        self.length = corpora.shape[0]

        self.sentences = {}

        self.passed = {}

    @classmethod
    def reload(cls, number: int = 1):

        cls.nb_process = number

    @classmethod
    def store_indices(cls, nb_paragraph, i: int, j: int):

        cls.indices[nb_paragraph] = {"i": i, "j": j}

    def increment(self, number: int = 1):

        ExtractRelatedSentences.nb_process += number

        if ExtractRelatedSentences.nb_process > self.length:
            pass
            # raise ValueError("The last paragraph is reached!")

    def decrement(self, number: int = 1):

        ExtractRelatedSentences.nb_process -= number

        if ExtractRelatedSentences.nb_process < 1:

            ExtractRelatedSentences.nb_process = 1

    def add_sentences(self, nb_paragraph: int, sentences: dict):

        self.sentences[nb_paragraph] = sentences

        if len(sentences["1"]) != len(sentences["2"]):

            raise LengthError(
                "The number of sentences in the two corpora must be equal!"
            )

    def add_passed(self, nb_paragraph: int, sentences: dict):

        self.sentences[nb_paragraph] = sentences

    def clear_sentences(self, nb_paragraph: int):

        sentences = self.sentences[nb_paragraph]

        clear = input(
            f"Are you sure you want to remove the following sentences!\
            \n\nOn {self.corpus_1}:\n{sentences['1']}\n\nOn {self.corpus_2}:\n{sentences['2']}\nYes(y), No(n) :"
        )

        if clear == "y":

            del sentences[nb_paragraph]

            print(f"Sentences at {nb_paragraph} was cleared!")

        elif clear == "n":

            print(f"Process aborted!")

        else:

            raise CommandError(f"You cannot take the command {clear}!")

    def get_sentences(self, nb_paragraph: int):

        return self.sentences[nb_paragraph]

    def unify(self, sentences: list, sentence: str, unification_marks: list):

        for mark in unification_marks:

            begin_mark = False

            end_mark = False

            for letter in sentences[-1]:

                if letter == mark[1]:

                    begin_mark = False

                elif letter == mark[0]:

                    begin_mark = True

            for letter in sentence:

                if letter == mark[1]:

                    end_mark = True

                    break

                else:

                    break

            if sentence != "" and sentence[0].islower():

                return True, " "

            if end_mark or begin_mark:

                space = " " if mark[2] else ""

                return True, space

        return False, " "

    def split(self, paragraph: str, ending_marks: list, unification_marks: list):

        prob_sentences = paragraph.strip()

        for mark in ending_marks:

            if isinstance(prob_sentences, list):

                new_sentences = prob_sentences.copy()

                counter = 0

                for s in prob_sentences:

                    if mark in s:

                        splits = new_sentences[counter].split(mark)

                        sentences = [
                            sentence.strip() + mark for sentence in splits[:-1]
                        ] + splits[-1:]

                        new_sentences = (
                            new_sentences[:counter]
                            + sentences
                            + new_sentences[counter + 1 :]
                        )

                        counter += len(sentences) - 1

                    else:

                        counter += 1

                prob_sentences = new_sentences

            else:

                if mark in prob_sentences:

                    splits = prob_sentences.split(mark)

                    prob_sentences = [
                        sentence.strip() + mark for sentence in splits[:-1]
                    ] + splits[-1:]

            new_sentences = []

            counter = 0

            for s in prob_sentences:

                unify, space = False, ""

                if counter != 0:

                    unify, space = self.unify(new_sentences, s, unification_marks)

                if s != "":

                    if not unify:

                        new_sentences.append(s)

                        counter += 1

                    else:

                        new_sentences[-1] = new_sentences[-1] + space + s

            prob_sentences = new_sentences

        return prob_sentences

    def __save(self, storage: str = "data/extractions/new_data/sent_extraction.txt"):

        with open(storage, "wb") as f:

            checkpoints = {
                "indices": ExtractRelatedSentences.indices,
                "nb_process": ExtractRelatedSentences.nb_process,
                "sentences": self.sentences,
                "passed": self.passed,
            }

            pickler = pickle.Pickler(f)

            pickler.dump(checkpoints)

    def save_data_frame(
        self,
        storage: str = "data/extractions/new_data/sent_extraction.txt",
        csv_file_path: str = "data/extractions/new_data/sent_extraction.csv",
        **kwargs,
    ):

        self.load(storage)

        data_frame = pd.DataFrame.from_dict(self.sentences[1], orient="columns")

        for i in range(2, self.length + 1):

            data_frame = pd.concat(
                (
                    data_frame,
                    pd.DataFrame.from_dict(self.sentences[i], orient="columns"),
                )
            )

        data_frame.rename(
            columns={"1": self.corpus_1, "2": self.corpus_2}, inplace=True
        )

        data_frame.to_csv(csv_file_path, index=False, **kwargs)

    def load(self, storage: str = "data/extractions/new_data/sent_extraction.txt"):

        with open(storage, "rb") as f:

            depickler = pickle.Unpickler(f)

            checkpoints = depickler.load()

            ExtractRelatedSentences.indices = checkpoints["indices"]

            ExtractRelatedSentences.nb_process = checkpoints["nb_process"]

            self.sentences = checkpoints["sentences"]

            self.passed = checkpoints["passed"]

    def preprocess(
        self,
        number: Union[int, None] = None,
        ending_marks: list = [".", " ?", " !"],
        unification_marks: List[Tuple] = [
            ("«", "»", True),
            ("(", ")", True),
            ("“", "”", True),
        ],
        cr: str = "r",
        cm1: str = "f",
        cm2: str = "j",
        cm3: str = "l",
        cmp1: str = "y",
        cmp2: str = "i",
        cmp3: str = "p",
        q: str = "q",
        start_at_last_indices: bool = False,
        i: int = 0,
        j: int = 0,
        auto_save: bool = True,
        storage: str = "data/extractions/new_data/sent_extraction.txt",
    ):

        line = number if not number is None else self.nb_process

        process_again = ""

        try:
            self.load(storage=storage)
        except:
            pass

        if line in set(self.sentences):

            process_again = input(
                f"You have already process the paragraph at line {line}.\nDo you want to modify from the processed sentences ? Yes(y), No(n):"
            )

        print(f"Preprocessing of paragraph at line {line}")

        if process_again == "n" or process_again == "":

            paragraph1 = str(self.corpora.loc[line - 1, self.corpus_1])

            paragraph2 = str(self.corpora.loc[line - 1, self.corpus_2])

            # let us separate the paragraphs by ending marks

            prob_sentences1 = self.split(paragraph1, ending_marks, unification_marks)

            prob_sentences2 = self.split(paragraph2, ending_marks, unification_marks)

        elif process_again == "y":

            prob_sentences1 = self.sentences[line]["1"]

            prob_sentences2 = self.sentences[line]["2"]

        else:

            raise CommandError(f"You cannot take the command {process_again}!")

        print("\n-----------\n-----------\n")

        print("Do you want to process the following sentences:\n")

        print(f"On {self.corpus_1}: ")

        [print(f"{i}: {sentence}") for i, sentence in enumerate(prob_sentences1)]

        print(f"\nOn {self.corpus_2}: ")

        [print(f"{i}: {sentence}") for i, sentence in enumerate(prob_sentences2)]

        print("\n-----------")

        process = input("Yes(y), Accept all (a) or No(n): ")

        cm = ""

        if process == "y":

            sentences = {"1": [], "2": []}

            passed = {"1": [], "2": []}

            last_sentences = {"1": "", "2": ""}

            if start_at_last_indices and line in set(self.indices):

                i = self.indices[line]["i"]

                j = self.indices[line]["j"]

            while i < len(prob_sentences1) and j < len(prob_sentences2):

                self.store_indices(line, i, j)

                sentence1 = sentence1_ = prob_sentences1[i]

                sentence2 = sentence2_ = prob_sentences2[j]

                if last_sentences["1"] != "":

                    sentence1_ = last_sentences["1"].strip() + " " + sentence1

                if last_sentences["2"] != "":

                    sentence2_ = last_sentences["2"] + " " + sentence2

                print(
                    f"\nThe current sentences are:\n{self.corpus_1} (index = {i}) : {sentence1_}\n{self.corpus_2} (index = {j}) : {sentence2_}"
                )

                cm = input(
                    f"Are they related together ?\n(index = {i}) : {sentence1_}\n{self.corpus_2} (index = {j}) : {sentence2_}\nQuit ({q}), Related ({cr}), Sentence 1 is uncompleted ({cm1}), Sentence 2 is uncompleted ({cm2}), The two sentences are uncompleted ({cm3}),\n Pass sentence 1 ({cmp1}), Pass sentence 2 ({cmp2}), Pass the two sentences ({cmp3}) :"
                )

                if cm == cr:

                    # clear the last sentences
                    last_sentences = {"1": "", "2": ""}

                    # add the sentences to the list of related sentences
                    sentences["1"].append(sentence1_.strip())

                    sentences["2"].append(sentence2_.strip())

                    # Pass to the next sentences
                    i += 1

                    j += 1

                elif cm == cm1:

                    # The first sentence is added to the last sentence 1
                    last_sentences["1"] += " " + sentence1

                    # Pass to the next sentence at corpus 1
                    i += 1

                elif cm == cm2:

                    # The second sentence is added to the last sentence 2
                    last_sentences["2"] += " " + sentence2

                    # Pass to the next sentence at corpus 2
                    j += 1

                elif cm == cm3:

                    # The two sentences are added to the last sentences
                    last_sentences["1"] += " " + sentence1

                    last_sentences["2"] += " " + sentence2

                    # Pass to the next sentences
                    i += 1

                    j += 1

                elif cm == cmp1:

                    # Clear the last sentence at corpus 1
                    last_sentences["1"] = ""

                    # Add the sentence 1 to the passed sentences at corpus 1
                    passed["1"].append(sentence1_)

                    # Pass to the next sentence at corpus 1
                    i += 1

                elif cm == cmp2:

                    # Clear the last sentence at corpus 2
                    last_sentences["2"] = ""

                    # Add the sentence 2 to the passed sentences at corpus 2
                    passed["2"].append(sentence2_)

                    # Pass to the next sentence at corpus 2
                    j += 1

                elif cm == cmp3:

                    # Clear the last sentences
                    last_sentences = {"1": "", "2": ""}

                    # Add the two sentences to the passed sentences
                    passed["1"].append(sentence1_)

                    passed["2"].append(sentence2_)

                    # Pass to the next sentences
                    i += 1

                    j += 1

                elif cm == q:

                    break

                else:

                    print(f"You cannot take the command {cm} ! Please retry again !")

                    print("\n-----------\n")

                    continue

                print("\nYou have stored the following sentences for the moment :\n")

                print(f"On {self.corpus_1} (length = {len(sentences['1'])}): ")

                print("\n".join(sentences["1"]))

                print(f"\nOn {self.corpus_2} (length = {len(sentences['2'])}): ")

                print("\n".join(sentences["2"]))

                print("\n-----------\n")

                print("You have passed the following sentences :\n")

                print(f"On {self.corpus_1} (length = {len(passed['1'])}): ")

                print("\n".join(passed["1"]))

                print(f"\nOn {self.corpus_2} (length = {len(passed['2'])}): ")

                print("\n".join(passed["2"]))

                print("\n------------------------------------")

                # add the sentences
                self.add_sentences(line, sentences)

                # add the passed sentences
                self.add_passed(line, passed)

                if auto_save:
                    # save the checkpoints
                    self.__save(storage)

            print("\nFinished!")

            # incrementing the number of processes
            self.increment()

            # add the passed sentences
            self.add_passed(line, passed)

            # add the sentences
            self.add_sentences(line, sentences)

            if auto_save:
                # save the checkpoints
                self.__save(storage)

        elif process == "a":

            sentences = {"1": prob_sentences1, "2": prob_sentences2}

            print("\nFinished!")

            # incrementing the number of processes
            self.increment()

            # add the sentences
            self.add_sentences(line, sentences)

            if auto_save:
                # save the checkpoints
                self.__save(storage)

        elif process == "n" or cm == "q":

            print(f"Process aborted!")

        else:

            raise CommandError(f"You cannot take the command {process}!")
