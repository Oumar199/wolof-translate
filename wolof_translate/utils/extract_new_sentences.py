from typing import *
import pandas as pd
import pickle
import re
import os


class NewSentenceExtraction:
    def __init__(
        self,
        text: Union[str, None] = None,
        sent_sep: str = ":",
        corpus_1: str = "wolof",
        corpus_2: str = "french",
        save_directory: str = "data/additional_documents/diagne_sentences/",
        checkpoint_name: str = "new_sentences",
    ):

        self.text = text

        self.corpus_1 = corpus_1

        self.corpus_2 = corpus_2

        self.sep = sent_sep

        self.groups = []

        self.index = 0

        self.save_directory = save_directory

        self.checkpoint = checkpoint_name

        self.extractions = {corpus_1: [], corpus_2: []}

    def __save(self):

        checkpoints = {
            # 'extractions': self.extractions,
            "index": self.index,
            # 'groups': self.groups
        }

        pd.DataFrame({"groups": self.groups}).to_csv(
            os.path.join(self.save_directory, "groups.csv"), index=False
        )

        pd.DataFrame(self.extractions).to_csv(
            os.path.join(self.save_directory, "extractions.csv"), index=False
        )

        with open(os.path.join(self.save_directory, self.checkpoint), "wb") as f:

            pickler = pickle.Pickler(f)

            pickler.dump(checkpoints)

    def sep_with_mark(self, group: str, mark: Union[str, None] = None):

        raise NotImplementedError

    def load(self):

        with open(os.path.join(self.save_directory, self.checkpoint), "rb") as f:

            depickler = pickle.Unpickler(f)

            checkpoints = depickler.load()

        try:

            self.extractions = pd.read_csv(
                os.path.join(self.save_directory, "extractions.csv")
            ).to_dict("list")

        except Exception:

            pass

        self.groups = pd.read_csv(os.path.join(self.save_directory, "groups.csv"))[
            "groups"
        ].to_list()

        self.index = checkpoints["index"]

    def add_groups(self, new_groups: list):

        self.groups += new_groups

        self.__save()

    def get_groups(self, stop_criterions: list = ["  ", "\n"], comparisons: list = []):

        assert not self.text is None

        i = 0

        a = 0

        g = 1

        while i < len(self.text):

            letter = self.text[i]

            if letter == self.sep:

                print(f"Extraction of group number {g}\n")

                b = i - 1  # index of letters before the current letter

                a = i + 1  # index of letters after the current letter

                corpus_1_s = []  # letters of the left sentence

                corpus_2_s = []  # letters of the right sentence

                stop = False

                for stop_cr in stop_criterions:

                    if self.text[b - len(stop_cr) + 1 : b + 1] == stop_cr:

                        stop = True

                while not stop:

                    corpus_1_s.append(self.text[b])

                    b -= 1

                    stop = False

                    for stop_cr in stop_criterions:

                        if self.text[b - len(stop_cr) + 1 : b + 1] == stop_cr:

                            stop = True

                stop = False

                for stop_cr in stop_criterions:

                    if self.text[a : a + len(stop_cr)] == stop_cr:

                        stop = True

                while not stop:

                    corpus_2_s.append(self.text[a])

                    a += 1

                    stop = False

                    for stop_cr in stop_criterions:

                        if self.text[a : a + len(stop_cr)] == stop_cr:

                            stop = True

                # reverse first sentence
                corpus_1_s.reverse()

                # add the sentences
                current_sentence = (
                    "".join(corpus_1_s).strip()
                    + f" {self.sep} "
                    + "".join(corpus_2_s).strip()
                )

                if "".join(corpus_1_s).strip() != "" and "".join(corpus_2_s) != "":

                    # verify if it is not already manually got
                    not_recuperated = True

                    for comparison in comparisons:

                        if current_sentence in comparison:

                            not_recuperated = False

                    # verify if it is not already in the extracted groups
                    for group in self.groups:

                        if current_sentence in group:

                            not_recuperated = False

                    if not_recuperated:

                        self.groups.append(current_sentence.strip())
                        # print(current_sentence)

                        g += 1

                        print("Successfully extracted !!\n")

                        print("-----------------\n")

                        i = a - 1

                        self.__save()

            i += 1

        # print("The groups were successfully recuperated !")

    def replace_groups(
        self,
        re_match: str,
        delete_re: Union[str, None] = None,
        n_replace_max: int = 1,
        load: bool = True,
        save: bool = False,
        manual_replace: bool = False,
        csv_file: str = "founded.csv",
        force_replace: bool = False,
    ):

        # we load the data
        if load:

            self.load()

        # find the groups matching the match regex
        founded = [
            (i, self.groups[i])
            for i in range(len(self.groups))
            if re.match(re_match, self.groups[i])
        ]

        print(
            f"Found groups matching the regular expression {re_match} are the followings:\n"
        )

        [print(f"- {f[1]}") for f in founded]

        print("\n----------------------\n")

        # if regex for deletion are provided we replace those that will be found with a max number of replace
        not_replaced = set()

        replaced = set()

        result = {}

        delete_re_ = input(
            "Do you want to change the deletion' regex expression -> provide one if yes or give empty string ('') if not : "
        )

        if delete_re_ != "":

            delete_re = delete_re_

        if not delete_re is None or manual_replace:

            for i in range(len(founded)):

                f = founded[i][1]

                index = founded[i][0]

                m_replace = "n"

                if not force_replace and manual_replace:

                    print(f"You will modify the following group:\n {f}")

                    m_replace = input(
                        f"\nDo you want to make a manual replacement of the group {f} -> Yes(y) or No(n). If you want to quit, press q!"
                    )

                    if m_replace == "q":

                        break

                    while not m_replace in ["y", "n"]:

                        replace_r = input(
                            f"You must provide a response between Yes(y), No(n)!"
                        )

                    if m_replace != "n":

                        print(
                            f"The manual modification of the group\n {f}\n is done in the following file: {csv_file}\n!If you want to provide multiple new groups please make them in different lines"
                        )

                        finish = "n"

                        pd.DataFrame({"to_modify": [f]}).to_csv(csv_file, index=False)

                        while finish == "n":

                            finish = input(
                                "Did you finish to replace -> No(n) if you didn't finish yet, click any another key if Yes(y) : "
                            )

                        f = pd.read_csv(csv_file)["to_modify"].to_list()

                    print("\n--------\n")

                if not delete_re is None and m_replace in ["n", ""]:

                    to_replace = set(re.findall(delete_re, f))

                    replace_r = None

                    for r in to_replace:

                        if force_replace:

                            f = f.replace(r, "", n_replace_max)

                            replaced.add(f)

                        else:

                            replace_r = input(
                                f"Do you want to replace the {r} string in the group:\n {f} ? Yes(y) or No(n). If you want to quit, press q!"
                            )

                            if m_replace == "q":

                                break

                            while not replace_r in ["y", "n"]:

                                replace_r = input(
                                    f"You must provide a response between Yes(y) and No(n)!"
                                )

                            if replace_r == "y":

                                f = f.replace(r, "", n_replace_max)

                                replaced.add(f)

                            else:

                                not_replaced.add(f)

                    if not replace_r is None and replace_r == "q":

                        break

                if isinstance(f, str):

                    f = [f.strip()]

                else:

                    f = [f_.strip() for f_ in f]

                try:

                    self.groups = self.groups[:index] + f + self.groups[index + 1 :]

                except IndexError:

                    self.groups = self.groups[:index] + f

                if len(f) > 1 and i != len(founded) - 1:

                    for j in range(i + 1, len(founded)):

                        founded[j] = (founded[j][0] + len(f) - 1, founded[j][1])

                result[index] = f

            if save:

                print("Final result:")

                [print(v) for r, v in result.items()]

                save_result = input("Do you want to save the result ? Yes(y) or No(n)")

                while not save_result in ["y", "n"]:

                    replace_r = input(
                        f"You must provide a response between Yes(y) or No(n) !"
                    )

                if save_result == "y":

                    self.__save()

        return {
            "founded": founded,
            "result": result,
            "replaced": replaced,
            "not_replaced": not_replaced,
        }

    def extraction_commands(
        self,
        add_end_mark_cmd: str = "a",
        pass_cmd: str = "p",
        add_end_mark_on_all: str = "l",
        add_upper_cmd: str = "u",
        add_upper_on_all: str = "o",
        sep_cmd: str = "_",
        quit_cmd: str = "q",
    ):

        # recuperate the current command
        cm = input(
            f"Choose one of the following commands: \n- {add_end_mark_cmd}+group_nb1,group_nb2:mark|group_nb3,group_nb4:mark|...(or group_nb1-group_nbn:mark) : To add end mark on specific groups\
                \n- {add_end_mark_on_all}+mark : To add end mark of all groups, \n- {add_upper_cmd}+group_nb1,group_nb2,group_nb3,group_nb4,...(or group_nb1-group_nbn) : To uppercase the first letter of specific groups\
                    \n- {add_upper_on_all} : To uppercase the first letter of all the groups\
                        \n- {pass_cmd} : To accept all of the groups\
                            \n- {quit_cmd} : To stop the process\
                                \n- You can combine all two commands by underscore {sep_cmd} excepted for the two last commands !"
        )

        cms = cm.split(sep_cmd)

        error = False

        if len(cms) == 2:

            p_cm = [cms[0].split("+")[0], cms[1].split("+")[0]]

            if pass_cmd in p_cm or quit_cmd in p_cm or sep_cmd in p_cm:

                print(
                    f"You cannot provide {pass_cmd}, {quit_cmd} or {sep_cmd} in combined commands !"
                )

                error = True

            elif (
                p_cm[0] in [add_end_mark_cmd, add_end_mark_on_all]
                and p_cm[1] in [add_upper_cmd, add_upper_on_all]
            ) or (
                p_cm[0] in [add_upper_cmd, add_upper_on_all]
                and p_cm[1] in [add_upper_cmd, add_upper_on_all]
            ):

                print(
                    "You cannot combine the same type of command: Type of commands are 'end mark' and 'upper'"
                )

        elif len(cms) == 1:

            if not cms[0].split("+")[0] in [
                add_end_mark_cmd,
                add_end_mark_on_all,
                add_upper_cmd,
                add_upper_on_all,
                pass_cmd,
                quit_cmd,
            ]:

                print("You didn't provide a right command ! Please retry")

                error = True

        else:

            print("You cannot provide more than 2 or 0 commands !")

        return cms, error

    def split_group(self, group: Union[list, str]):
        # we base on the colon critter to split the groups

        if isinstance(group, str):

            group = [group]

        sents = {self.corpus_1: [], self.corpus_2: []}

        for g in group:

            splits = g.split(":")

            middle = len(splits) // 2

            cp1_corpus = "".join(splits[:middle])

            cp2_corpus = "".join(splits[middle:])

            sents[self.corpus_1].append(cp1_corpus.strip())

            sents[self.corpus_2].append(cp2_corpus.strip())

        return sents

    def add_end_mark(self, batch: dict, command: str):

        cm = command

        # recuperate the marks with groups and apply the transformations
        tfs = cm.split("|")

        for tf in tfs:

            if "-" in tf:

                groups = tf.split(":")[0].split("-")

                groups = list(range(int(groups[0]), int(groups[1]) + 1))

            else:

                groups = [int(nb) for nb in tf.split(":")[0].split(",")]

            mark = tf.split(":")[1]

            for nb in groups:

                batch[self.corpus_1][nb - 1] += mark

                batch[self.corpus_2][nb - 1] += mark

        return batch

    def add_upper(self, batch: dict, command: str):

        cm = command

        # recuperate the marks with groups and apply the transformations
        tfs = cm.split("|")

        for tf in tfs:

            # recuperate the marks with groups and apply the transformations
            if "-" in tf:

                groups = tf.split("-")

                groups = list(range(int(groups[0]), int(groups[1]) + 1))

            else:

                groups = [int(nb) for nb in tf.split(",")]

            for nb in groups:

                batch[self.corpus_1][nb - 1] = (
                    batch[self.corpus_1][nb - 1][0].upper()
                    + batch[self.corpus_1][nb - 1][1:]
                )

                batch[self.corpus_2][nb - 1] = (
                    batch[self.corpus_2][nb - 1][0].upper()
                    + batch[self.corpus_2][nb - 1][1:]
                )

        return batch

    def inner_command(self, batch: dict):

        cp1_sents = batch[self.corpus_1]

        cp2_sents = batch[self.corpus_2]

        for i in range(0, len(batch[self.corpus_1])):

            cp1_sent = cp1_sents[i]

            cp2_sent = cp2_sents[i]

            if re.match(".*Mark\[.*\].*", cp2_sent):

                mark = re.findall("Mark\[.*\]", cp2_sent)[0]

                mark = mark.replace("Mark[", "").replace("]", "")

                cp1_sent = cp1_sent + mark

                cp2_sent = re.sub("Mark\[.*\]", "", cp2_sent, 1) + mark

            if re.match(".*Upper", cp2_sent):

                cp1_sent = cp1_sent[0].upper() + cp1_sent[1:]

                cp2_sent = cp2_sent[0].upper() + re.sub("Upper", "", cp2_sent, 1)[1:]

            cp1_sents[i] = cp1_sent

            cp2_sents[i] = cp2_sent

        batch[self.corpus_1] = cp1_sents

        batch[self.corpus_2] = cp2_sents

        return batch

    def extract_sentences(
        self,
        group_range: Union[tuple, None] = None,
        add_end_mark_cmd: str = "a",
        pass_cmd: str = "p",
        add_end_mark_on_all: str = "l",
        add_upper_cmd: str = "u",
        add_upper_on_all: str = "o",
        sep_cmd: str = "_",
        quit_cmd: str = "q",
        batch_size: int = 30,
        load: bool = True,
        save: bool = False,
        csv_file: str = "batch.csv",
        last_checkpoint: bool = True,
    ):

        # we load the data
        if load:

            self.load()

        # the group range is equal to a tuple containing the last saved index and the index of the last element in the list of groups
        # indices if nothing is given
        if last_checkpoint:

            if group_range is None:
                group_range = (self.index, len(self.groups) - 1)

        else:

            raise ValueError(
                "You must provide a group range if last checkpoint is to False !"
            )

        # change the number of displayed lines
        pd.options.display.max_rows = batch_size

        groups = self.groups[group_range[0] : group_range[1] + 1]

        # initialize the sub corpora
        sub_corpora = {self.corpus_1: [], self.corpus_2: []}

        i = 0

        # for each batch we will add the groups in a csv file and take a command
        for b in range(0, len(groups), batch_size):

            # recuperate a batch
            batch_ = groups[b : b + batch_size]

            # recuperate the index
            self.index += len(batch_)

            # split each group into two sentences and transform the obtained dictionary to a DataFrame
            batch = self.split_group(batch_)

            pd.DataFrame(batch).to_csv(csv_file, index=False)

            print(
                f"Which of the groups of batch number {i+1} do you consider to be complete sentences (see the file {csv_file}) ?"
            )

            error = False

            cms = []

            try:

                cms, error = self.extraction_commands(
                    add_end_mark_cmd,
                    pass_cmd,
                    add_end_mark_on_all,
                    add_upper_cmd,
                    add_upper_on_all,
                    sep_cmd,
                    quit_cmd,
                )

            except Exception:

                print("You didn't provide a right group number !")

                error = True

            while error:

                error = False

                try:

                    cms, error = self.extraction_commands(
                        add_end_mark_cmd,
                        pass_cmd,
                        add_end_mark_on_all,
                        add_upper_cmd,
                        add_upper_on_all,
                        sep_cmd,
                        quit_cmd,
                    )

                except IndexError:

                    print("You didn't provide a right group number !")

                    error = True

            # recuperate the batch
            batch = pd.read_csv(csv_file).to_dict("list")

            # add corrections
            batch = self.inner_command(batch)

            cm_type = ""

            quit_ = "n"

            for cm in cms:

                cm_type = cm.split("+")[0]

                if cm_type == add_end_mark_cmd:

                    batch = self.add_end_mark(batch, cm.split("+")[1])

                elif cm_type == add_end_mark_on_all:

                    mark = cm.split("+")[1]

                    batch = self.add_end_mark(
                        batch,
                        ",".join(
                            [str(nb) for nb in range(1, len(batch[self.corpus_1]) + 1)]
                        )
                        + f":{mark}",
                    )

                elif cm_type == add_upper_cmd:

                    batch = self.add_upper(batch, cm.split("+")[1])

                elif cm_type == add_upper_on_all:

                    batch = self.add_upper(
                        batch,
                        ",".join(
                            [str(nb) for nb in range(1, len(batch[self.corpus_1]) + 1)]
                        ),
                    )

                elif cm_type == quit_cmd:

                    quit_ = input("Are you sure you want to quit: Yes(y) or No(n)")

                    while not quit_ in ["y", "n"]:

                        quit_ = input("Are you sure you want to quit: Yes(y) or No(n)")

                    if quit_ == "y":

                        break

                print("\nBatch result")

                print(pd.DataFrame(batch).head(batch_size))

            print("\n--------------------\n\n")

            # add the batch to the sub corpora
            sub_corpora[self.corpus_1].extend(batch[self.corpus_1])

            sub_corpora[self.corpus_2].extend(batch[self.corpus_2])

            if cm_type == quit_cmd and quit_ == "y":

                break

            else:

                if save:

                    save_ = input("Do you want to save the result ? Yes(y) or No(n)")

                    while not save_ in ["y", "n"]:

                        save_ = input(
                            "Do you want to save the result ? Yes(y) or No(n)"
                        )

                    if save_ == "y":

                        self.extractions[self.corpus_1].extend(batch[self.corpus_1])

                        self.extractions[self.corpus_2].extend(batch[self.corpus_2])

                        self.__save()

            i += 1

        print("Finished !")

    def remove_duplicated_sentences(self, save: bool = False):

        # we load the data
        self.load()

        # use pandas to delete the duplicated rows
        extractions = pd.DataFrame(self.extractions)

        extractions.drop_duplicates(inplace=True)

        self.extractions = extractions.to_dict("list")

        # save the sentences
        if save:

            self.__save()
